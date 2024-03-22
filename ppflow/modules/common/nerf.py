"""
NERF!
Note that this was designed with compatibility with biotite, NOT biopython!
These two packages use different conventions for where NaNs are placed in dihedrals

References:
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
https://www.biotite-python.org/examples/gallery/structure/peptide_assembly.html
"""
import os
from functools import cached_property
from typing import *

import numpy as np
import torch

N_CA_LENGTH = 1.46  # Check, approxiamtely right
CA_C_LENGTH = 1.53  # Check, approximately right
C_N_LENGTH = 1.34  # Check, approximately right
O_C_LENGTH = 1.23

N_CA_C_ANGLE = 1.93
O_C_N_ANGLE = 2.14
CA_C_N_ANGLE = 2.02
C_N_CA_ANGLE = 2.12

O_C_N_CA_DIHED = 0.0

# Taken from initial coords from 1CRN, which is a THR
N_INIT = np.array([-0.525, 1.363, 0.0])
CA_INIT = np.array([0.0, 0.0, 0.0])
C_INIT = np.array([1.526, -0.0, -0.0])
O_CA_RELATIVE = np.array([2.153, -1.062, 0.0])

class NERFBuilder:
    """
    Builder for NERF
    """

    def __init__(
        self,
        phi_dihedrals: np.ndarray,
        psi_dihedrals: np.ndarray,
        omega_dihedrals: np.ndarray,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,  # 0C:1N distance
        bond_angle_n_ca: Union[float, np.ndarray] = 2.1125572,
        bond_angle_ca_c: Union[float, np.ndarray] = 1.9400245,  # aka tau
        bond_angle_c_n: Union[float, np.ndarray] = 2.0276928,
        init_coords: np.ndarray = [N_INIT, CA_INIT, C_INIT],
    ) -> None:
        self.use_torch = False
        if any(
            [
                isinstance(v, torch.Tensor)
                for v in [phi_dihedrals, psi_dihedrals, omega_dihedrals]
            ]
        ):
            self.use_torch = True

        self.phi = phi_dihedrals.squeeze()
        self.psi = psi_dihedrals.squeeze()
        self.omega = omega_dihedrals.squeeze()

        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the C --> N bond
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
        }
        self.init_coords = [c.squeeze() for c in init_coords]
        assert (
            len(self.init_coords) == 3
        ), f"Requires 3 initial coords for N-Ca-C but got {len(self.init_coords)}"
        assert all(
            [c.size == 3 for c in self.init_coords]
        ), "Initial coords should be 3-dimensional"

    @cached_property
    def cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Build out the molecule"""
        retval = self.init_coords.copy()
        if self.use_torch:
            retval = [torch.tensor(x, requires_grad=True) for x in retval]

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        phi = self.phi[1:]
        psi = self.psi[:-1]
        omega = self.omega[:-1]
        dih_angles = (
            torch.stack([psi, omega, phi])
            if self.use_torch
            else np.stack([psi, omega, phi])
        ).T
        assert (
            dih_angles.shape[1] == 3
        ), f"Unexpected dih_angles shape: {dih_angles.shape}"

        for i in range(dih_angles.shape[0]):
            # for i, (phi, psi, omega) in enumerate(
            #     zip(self.phi[1:], self.psi[:-1], self.omega[:-1])
            # ):
            dih = dih_angles[i]
            # Procedure for placing N-CA-C
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            for j, bond in enumerate(self.bond_lengths.keys()):
                coords = place_dihedral(
                    retval[-3],
                    retval[-2],
                    retval[-1],
                    bond_angle=self._get_bond_angle(bond, i),
                    bond_length=self._get_bond_length(bond, i),
                    torsion_angle=dih[j],
                    use_torch=self.use_torch,
                )
                retval.append(coords)

        if self.use_torch:
            return torch.stack(retval)
        return np.array(retval)

    @cached_property
    def centered_cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Returns the centered coords"""
        means = self.cartesian_coords.mean(axis=0)
        return self.cartesian_coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond distance"""
        v = self.bond_lengths[bond]
        if isinstance(v, float):
            return v
        return v[idx]

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond angle"""
        v = self.bond_angles[bond]
        if isinstance(v, float):
            return v
        return v[idx]


def place_dihedral(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_angle: float,
    bond_length: float,
    torsion_angle: float,
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.shape == b.shape == c.shape
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 3

    if not use_torch:
        unit_vec = lambda x: x / np.linalg.norm(x, axis=-1)
        cross = lambda x, y: np.cross(x, y, axis=-1)
    else:
        ensure_tensor = (
            lambda x: torch.tensor(x, requires_grad=False).to(a.device)
            if not isinstance(x, torch.Tensor)
            else x.to(a.device)
        )
        a, b, c, bond_angle, bond_length, torsion_angle = [
            ensure_tensor(x) for x in (a, b, c, bond_angle, bond_length, torsion_angle)
        ]
        unit_vec = lambda x: x / torch.linalg.norm(x, dim=-1, keepdim=True)
        cross = lambda x, y: torch.linalg.cross(x, y, dim=-1)

    ab = b - a
    bc = unit_vec(c - b)
    n = unit_vec(cross(ab, bc))
    nbc = cross(n, bc)

    if not use_torch:
        m = np.stack([bc, nbc, n], axis=-1)
        d = np.stack(
            [
                -bond_length * np.cos(bond_angle),
                bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
                bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
            ],
            axis=a.ndim - 1,
        )
        d = m.dot(d)
    else:
        m = torch.stack([bc, nbc, n], dim=-1)
        d = torch.stack(
            [
                -bond_length * torch.cos(bond_angle),
                bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
                bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle),
            ],
            dim=a.ndim - 1,
        ).type(m.dtype)
        d = torch.matmul(m, d).squeeze()

    return d + c



def nerf_build_batch_bb4(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
    bond_angle_n_ca_c: Union[float, torch.Tensor] = N_CA_C_ANGLE,  # theta1
    bond_angle_ca_c_n: Union[float, torch.Tensor] = CA_C_N_ANGLE,  # theta2
    bond_angle_c_n_ca: Union[float, torch.Tensor] = C_N_CA_ANGLE,  # theta3
    bond_len_n_ca: Union[float, torch.Tensor] = N_CA_LENGTH,
    bond_len_ca_c: Union[float, torch.Tensor] = CA_C_LENGTH,
    bond_len_c_n: Union[float, torch.Tensor] = C_N_LENGTH,  # 0C:1N distance
    dihed_o_c_n_ca: Union[float, torch.Tensor] = O_C_N_CA_DIHED,
    bond_angle_o_c_n: Union[float, torch.Tensor] = O_C_N_ANGLE,
    bond_len_o_c: Union[float, torch.Tensor] = O_C_LENGTH, 
    init_pos = torch.tensor(np.array([N_INIT, CA_INIT, C_INIT])),
    final_o_c_ca_n = 225 / 180 * np.pi) -> torch.Tensor:
    """
    Build out a batch of phi, psi, omega values. Returns the 3D coordinates
    in Cartesian space with the shape (batch, length * 4, 3). Here, length is
    multiplied by 3 because for each backbone, there are coordinates for the
    N-CA-C=O atoms.
    """
    assert phi.ndim == psi.ndim == omega.ndim == 2  # batch, seq
    assert phi.shape == psi.shape == omega.shape
    batch = phi.shape[0]
    
    coords = (
        torch.clone(init_pos)
        .repeat(batch, 1, 1)
        .to(phi.device)
    )
    assert coords.shape == (batch, 3, 3), f"Mismatched shape: {coords.shape}"

    # perform broadcasting of bond lengths
    ensure_tensor = (
        lambda x: torch.tensor(x, requires_grad=False).expand(phi.shape)
        if isinstance(x, float)
        else x
    )
    bond_len_n_ca = ensure_tensor(bond_len_n_ca)
    bond_len_ca_c = ensure_tensor(bond_len_ca_c)
    bond_len_c_n = ensure_tensor(bond_len_c_n)
    bond_len_o_c = ensure_tensor(bond_len_o_c)

    bond_angle_c_n_ca = ensure_tensor(bond_angle_c_n_ca)
    bond_angle_ca_c_n = ensure_tensor(bond_angle_ca_c_n)
    bond_angle_n_ca_c = ensure_tensor(bond_angle_n_ca_c)
    bond_angle_o_c_n = ensure_tensor(bond_angle_o_c_n)

    dihed_o_c_n_ca = ensure_tensor(dihed_o_c_n_ca)
    final_o_c_ca_n = ensure_tensor(final_o_c_ca_n)

    phi = phi[:, 1:]
    psi = psi[:, :-1]
    omega = omega[:, 1:]

    assert phi.shape == psi.shape == omega.shape
    o_coords = []
    for i in range(phi.shape[1]):
        # Place the C-N
        n_coord = place_dihedral(
            coords[:, -3, :],  
            coords[:, -2, :],
            coords[:, -1, :],
            bond_angle=bond_angle_ca_c_n[:, i].unsqueeze(1),
            bond_length=bond_len_c_n[:, i].unsqueeze(1),
            torsion_angle=psi[:, i].unsqueeze(1),
            use_torch=True,
        )

        # Place the N-CA
        ca_coord = place_dihedral(
            coords[:, -2, :],
            coords[:, -1, :],
            n_coord,
            bond_angle=bond_angle_c_n_ca[:, i].unsqueeze(1),
            bond_length=bond_len_n_ca[:, i].unsqueeze(1),
            torsion_angle=omega[:, i].unsqueeze(1),
            use_torch=True,
        )

        # Place the CA-C
        c_coord = place_dihedral(
            coords[:, -1, :],
            n_coord,
            ca_coord,
            bond_angle=bond_angle_n_ca_c[:, i].unsqueeze(1),
            bond_length=bond_len_ca_c[:, i].unsqueeze(1),
            torsion_angle=phi[:, i].unsqueeze(1),
            use_torch=True,
        )
        
        
        if i >= 1:
            o_coord = place_dihedral(
                ca_coord,
                n_coord,
                coords[:, -1, :],
                bond_angle=bond_angle_o_c_n[:, i-1].unsqueeze(1),
                bond_length=bond_len_o_c[:, i-1].unsqueeze(1),
                torsion_angle=dihed_o_c_n_ca[:, i-1].unsqueeze(1),
                use_torch=True,
            )
            
        else:
            o_coord = place_dihedral(
                ca_coord,
                n_coord,
                coords[:, -1, :],
                bond_angle=ensure_tensor(O_C_N_ANGLE)[:,0].unsqueeze(1),
                bond_length=ensure_tensor(O_C_LENGTH)[:,0].unsqueeze(1),
                torsion_angle=ensure_tensor(0.)[:,0].unsqueeze(1),
                use_torch=True,
            )

        o_coords.append(o_coord)
        coords = torch.cat(
                [coords, n_coord.unsqueeze(1), ca_coord.unsqueeze(1), c_coord.unsqueeze(1)],
                dim=1,
                )

    coords = coords.reshape(batch, -1, 3, 3)    
    final_pos = (
            torch.clone(coords)
    )
    o_coord = final_pos[:, -1, 1, : ] + torch.tensor(O_CA_RELATIVE).to(phi.device).unsqueeze(0)
    o_coords.append(o_coord)
    o_coords = torch.stack(o_coords, dim=1)
    coords_bb4 = torch.cat(
        [coords, o_coords.unsqueeze(-2)],
        dim=-2,
    )

    return coords.float(), coords_bb4.float()


def nerf_build_batch(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
    bond_angle_n_ca_c: Union[float, torch.Tensor] = N_CA_C_ANGLE,  # theta1
    bond_angle_ca_c_n: Union[float, torch.Tensor] = CA_C_N_ANGLE,  # theta2
    bond_angle_c_n_ca: Union[float, torch.Tensor] = C_N_CA_ANGLE,  # theta3
    bond_len_n_ca: Union[float, torch.Tensor] = N_CA_LENGTH,
    bond_len_ca_c: Union[float, torch.Tensor] = CA_C_LENGTH,
    bond_len_c_n: Union[float, torch.Tensor] = C_N_LENGTH,  # 0C:1N distance
    init_pos = torch.tensor(np.array([N_INIT, CA_INIT, C_INIT]))
    ):
    """
    Build out a batch of phi, psi, omega values. Returns the 3D coordinates
    in Cartesian space with the shape (batch, length * 4, 3). Here, length is
    multiplied by 3 because for each backbone, there are coordinates for the
    N-CA-C=O atoms.
    """
    assert phi.ndim == psi.ndim == omega.ndim == 2  # batch, seq
    assert phi.shape == psi.shape == omega.shape
    batch = phi.shape[0]
    
    coords = (
        torch.clone(init_pos)
        .repeat(batch, 1, 1)
        .to(phi.device)
    )
    assert coords.shape == (batch, 3, 3), f"Mismatched shape: {coords.shape}"

    # perform broadcasting of bond lengths
    ensure_tensor = (
        lambda x: torch.tensor(x, requires_grad=False).expand(phi.shape)
        if isinstance(x, float)
        else x
    )
    bond_len_n_ca = ensure_tensor(bond_len_n_ca)
    bond_len_ca_c = ensure_tensor(bond_len_ca_c)
    bond_len_c_n = ensure_tensor(bond_len_c_n)

    bond_angle_c_n_ca = ensure_tensor(bond_angle_c_n_ca)
    bond_angle_ca_c_n = ensure_tensor(bond_angle_ca_c_n)
    bond_angle_n_ca_c = ensure_tensor(bond_angle_n_ca_c)

    phi = phi[:, 1:]
    psi = psi[:, :-1]
    omega = omega[:, 1:]

    assert phi.shape == psi.shape == omega.shape
    for i in range(phi.shape[1]):
        # Place the C-N
        n_coord = place_dihedral(
            coords[:, -3, :],  
            coords[:, -2, :],
            coords[:, -1, :],
            bond_angle=bond_angle_ca_c_n[:, i].unsqueeze(1),
            bond_length=bond_len_c_n[:, i].unsqueeze(1),
            torsion_angle=psi[:, i].unsqueeze(1),
            use_torch=True,
        )

        # Place the N-CA
        ca_coord = place_dihedral(
            coords[:, -2, :],
            coords[:, -1, :],
            n_coord,
            bond_angle=bond_angle_c_n_ca[:, i].unsqueeze(1),
            bond_length=bond_len_n_ca[:, i].unsqueeze(1),
            torsion_angle=omega[:, i].unsqueeze(1),
            use_torch=True,
        )

        # Place the CA-C
        c_coord = place_dihedral(
            coords[:, -1, :],
            n_coord,
            ca_coord,
            bond_angle=bond_angle_n_ca_c[:, i].unsqueeze(1),
            bond_length=bond_len_ca_c[:, i].unsqueeze(1),
            torsion_angle=phi[:, i].unsqueeze(1),
            use_torch=True,
        )
                    

        coords = torch.cat(
                [coords, n_coord.unsqueeze(1), ca_coord.unsqueeze(1), c_coord.unsqueeze(1)],
                dim=1,
                )

    coords = coords.reshape(batch, -1, 3, 3)    


    return coords.float()


def main():
    """On the fly testing"""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    source = PDBFile.read(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
    )
    source_struct = source.get_structure()
    # print(source_struct[0])
    phi, psi, omega = [torch.tensor(x) for x in struc.dihedral_backbone(source_struct)]

    builder = NERFBuilder(phi, psi, omega)
    print(builder.cartesian_coords)
    print(builder.cartesian_coords.shape)


if __name__ == "__main__":
    main()
