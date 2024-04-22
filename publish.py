import os

import zipapps

zipapps.create_app(
    "./morebuiltins",
    main="morebuiltins",
    output="./morebuiltins.pyz",
    compressed=True,
)
os.system("flit publish")
