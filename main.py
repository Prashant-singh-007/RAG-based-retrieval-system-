import os
import sys
from src.rag import main
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    main()