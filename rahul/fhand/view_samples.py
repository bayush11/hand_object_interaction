from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
import argparse

from utils.fh_utils import *


def show_training_samples(base_path, version, num2show=None, render_mano=False):
    if render_mano:
        from utils.model import HandModel, recover_root, get_focal_pp, split_theta

    if num2show == -1:
        num2show = db_size('training') # show all

    # load annotations
    db_data_anno = load_db_annotation(base_path, 'training')

    # iterate over all samples
    for idx in range(db_size('training')):
        if idx >= num2show:
            break

        # load image and mask
        img = read_img(idx, base_path, 'training', version)
        msk = read_msk(idx, base_path)
        
        db_data_anno = list(db_data_anno)
        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        print(xyz[:, 1])
        xyz[0][0] = 0.5
        # print(mano[0])
        # mano[0] = [
        #         # 1.57, 1.57, 1.57, # first three are for wrist (global orientation)
        #         0,  0, 0, # first three are for wrist (global orientation)
        #         0, 0, -np.pi/8, # pointer finger lower
        #         0, 0, -np.pi/4,   # pointer finger middle
        #         0, 0, -np.pi/8,  # pointer finger upper
        #         0, 0, -np.pi/5, # middle finger lower
        #         0, 0, -np.pi/4,  # middle finger middle
        #         0, 0, -np.pi/8,  # middle finger upper
        #         0, 0, -np.pi/4,  # pinky finger lower
        #         0, 0, -np.pi/5,  # pinky finger middle
        #         0, 0, -np.pi/10,  # pinky finger upper
        #         0, 0, -np.pi/4,  # ring finger lower
        #         0, 0, -np.pi/5,  # ring finger middle
        #         0, 0, -np.pi/6, # ring finger upper
        #         0, 0, -np.pi/5, # thumb finger lower
        #         0, 0, -np.pi/5, # thumb finger middle
        #         0, 0, -np.pi/5,  # thumb finger upper
        #         0,0,0,0,0,0,0,0,0,0,
        #         1.19728488e+02,  1.16920922e+02, 578] # global translation xy and scale
        # mano[0][60] = -9
        # print(mano[0].shape)
        joints = mano[0][3:48].reshape(15,3)
        joints = joints[:, 2]
        for i in joints:
            print(i, end=', ')
        # shadowhand = np.linalg.norm(joints, axis=0)
        # print(shadowhand)
        # print(joints)
        uv = projectPoints(xyz, K)

        # render an image of the shape
        msk_rendered = None
        if render_mano:
            # split mano parameters
            poses, shapes, uv_root, scale = split_theta(mano)
            print(uv_root)
            focal, pp = get_focal_pp(K)
            xyz_root = recover_root(uv_root, scale, focal, pp)
            print(xyz_root)

            # set up the hand model and feed hand parameters
            renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
            renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
            msk_rendered = renderer.render(K, img_shape=img.shape[:2])

        # show
        fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(121)
        # ax1.imshow(img)
        ax2.imshow(msk if msk_rendered is None else msk_rendered)
        # plot_hand(ax1, uv, order='uv')
        # plot_hand(ax2, uv, order='uv')
        # ax1.axis('off')
        ax2.axis('off')
        plt.show()
        print("\n\n\n")


def show_eval_samples(base_path, num2show=None):
    if num2show == -1:
        num2show = db_size('evaluation') # show all

    for idx in range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--show_eval', action='store_true',
                        help='Shows samples from the evaluation split if flag is set, shows training split otherwise.')
    parser.add_argument('--mano', action='store_true',
                        help='Enables rendering of the hand if mano is available. See README for details.')
    parser.add_argument('--num2show', type=int, default=-1,
                        help='Number of samples to show. ''-1'' defaults to show all.')
    parser.add_argument('--sample_version', type=str, default=sample_version.gs,
                        help='Which sample version to use when showing the training set.'
                             ' Valid choices are %s' % sample_version.valid_options())
    args = parser.parse_args()

    # check inputs
    msg = 'Invalid choice: ''%s''. Must be in %s' % (args.sample_version, sample_version.valid_options())
    assert args.sample_version in sample_version.valid_options(), msg

    if args.show_eval:
        """ Show some evaluation samples. """
        show_eval_samples(args.base_path,
                          num2show=args.num2show)

    else:
        """ Show some training samples. """
        show_training_samples(
            args.base_path,
            args.sample_version,
            num2show=args.num2show,
            render_mano=args.mano
        )

