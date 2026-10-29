import os
import pybullet_data

root = pybullet_data.getDataPath()
for name in sorted(os.listdir(root)):
    if name.endswith(".urdf"):
        print(name)
print("\nFolders:", [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))])