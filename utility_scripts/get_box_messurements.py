import open3d as o3d
import numpy as np
import sys

def print_bbox_dimensions_meters(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        print(f"[Fehler] Punktwolke enthält keine Punkte: {ply_path}")
        return

    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = np.array(aabb.min_bound)
    max_bound = np.array(aabb.max_bound)
    center = (min_bound + max_bound) / 2
    half_dims = (max_bound - min_bound) / 2

    # Falls Maße sehr groß sind (>10), vermutlich in mm, Umrechnung in Meter:
    if np.any(half_dims > 10):
        print("[Hinweis] Große Maße erkannt – evtl. sind die Einheiten in Millimeter. Umrechne in Meter...")
        half_dims /= 1000
        center /= 1000

    print(f"Datei: {ply_path}")
    print(f"Zentrum der Bounding Box (in Metern): {center}")
    print(f"Halbmaße (Width/2, Depth/2, Height/2) in Metern: {half_dims}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Benutzung: python get_bbox_simple.py <Pfad_zur_ply_datei>")
        sys.exit(1)

    ply_file = sys.argv[1]
    print_bbox_dimensions_meters(ply_file)
