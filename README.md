# Find and Count Checkers

to run the code need to have OpenCV installed

`usage: python warp_and_find_checkers.py <input_path> <output_path>`

question 1: Depends a lot on the angle of the board images, if the angle is too high the distortion of the Perspective warp will also be large, besides of that the occlusion of the checkers will also happens and this will cause problems like the vertical image example

question 2: In the cases we have stressed angles of the photos and occlusions of the checkers. If the checkers colors were the same as every case we could use Color detection and detect the regions that have the checkers or we could train an Deep Learning Object Detection model to detect more extreme cases.

question 3: I would create a Color Detector along side the circle detector, if the color of the checker matches with the other checkers so its one player the same for the another player. Also would use Opening and Closing operations to remove the noises.
