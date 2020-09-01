import cv2
import numpy as np
import os

# package imports
from initial_conditions import unidimensional_lattice


np.random.seed(60)


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    lat = unidimensional_lattice(J=1, alpha=1e-1, dt=1e-4, icond='rand')

    mkdir('img')
    filenames = []
    for i in range(100):
        for _ in range(50000):
            lat.time_step()
        
        filenames.append(f'img/example-{i}.png')
        lat.plot(S=10, filename=filenames[-1])
            
    print( np.array(lat.get_M()) )
    if(len(filenames) > 0):
        generate_avi(filenames, deleteAll=True)


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------
def generate_avi(filenames, fps=20, vidname='example', deleteAll=False):
    """Takes a list of images' filenames and makes a video out of them.
    """
    frame = cv2.imread(filenames[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter("%s.avi" % vidname, 0, fps, (width,height))

    for filename in filenames:
        video.write(cv2.imread(filename))

    cv2.destroyAllWindows()
    video.release()
    
    if(deleteAll is True):
        # remove images
        for filename in filenames:
            os.remove(filename)


def mkdir(dir):
    """Create a directory.  If it already exists do nothing

    Parameters
    ----------
    dir: str
        Name of the directory
    """
    if(os.path.exists(dir) is False and dir not in ['']):
        os.mkdir(dir)

# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()