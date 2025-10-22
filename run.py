#!/usr/bin/env python
"""
Compatibility script to run the pipeline from the project root.
This allows running the project without installing it as a package.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main
from tasacion_portal.main import main

if __name__ == "__main__":
    main()
