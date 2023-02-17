IMG_KEYS = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT','CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
pos_map = [2, 3, 1, 5, 6, 4]
import matplotlib.pyplot as plt
def draw(imgs, dump_file):
    plt.figure(figsize=(24, 8))

    for i, img in enumerate(imgs):
        # Draw camera views
        fig_idx = pos_map[i]
        plt.subplot(2, 3, fig_idx)

        # Set camera attributes
        plt.title(IMG_KEYS[i])
        plt.axis('off')
        plt.xlim(0, 1600)
        plt.ylim(900, 0)

        # Draw images
        plt.imshow(img)
    plt.tight_layout(w_pad=0, h_pad=2)
    plt.savefig(dump_file)