def apply_augment(aug, img):
    aug_img = aug.augment(image=img)
    plt.imshow(aug_img)
    plt.axis('off')