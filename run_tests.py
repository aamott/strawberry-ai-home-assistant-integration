import sys
import os
import unittest

def main():
    # Add the current directory to sys.path so we can import custom_components
    # This assumes the script is run from the ha-integration directory or its parent
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Also add the parent directory if running from inside tests/ (just in case)
    sys.path.insert(0, os.path.join(current_dir, ".."))

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(current_dir, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    main()
