import sys
sys.path.append('/home/bigger/projects/five_in_a_row')
from inference.client import require_fit

if __name__ == '__main__':
    require_fit(200, 2000)
