"""Check the heightfield maps h[ix, iy] -> world (x = ix*DX, y = (iy-(ny-1)/2)*DX)."""
import numpy as np, pybullet as pb
from rover.terrain import DX

def test_axes():
    nx, ny = 300, 60
    h = np.zeros((nx, ny))
    h += np.linspace(0, 0.3, nx)[:, None]          # ramp along x: 0 -> 0.3 m
    h[:, : ny // 2] += 0.05                          # step: the -y half is 5 cm higher
    c = pb.connect(pb.DIRECT)
    shape = pb.createCollisionShape(shapeType=pb.GEOM_HEIGHTFIELD, meshScale=[DX, DX, 1],
                                    heightfieldData=h.T.ravel().tolist(),
                                    numHeightfieldRows=nx, numHeightfieldColumns=ny, physicsClientId=c)
    pb.createMultiBody(0, shape, basePosition=[(nx - 1) * DX / 2, 0, 0.5 * (h.max() + h.min())], physicsClientId=c)
    errs = []
    for x in (0.5, 1.5, 2.5):
        for y in (-0.2, 0.2):
            hit = pb.rayTest([x, y, 2], [x, y, -2], physicsClientId=c)[0][3][2]
            ix, iy = int(round(x / DX)), int(round(y / DX + (ny - 1) / 2))
            errs.append(abs(hit - h[ix, iy]))
            print(f"x={x} y={y}: ray z={hit:.4f}  grid h={h[ix, iy]:.4f}")
    pb.disconnect(c)
    assert max(errs) < 2e-3, errs

if __name__ == "__main__":
    test_axes(); print("OK")
