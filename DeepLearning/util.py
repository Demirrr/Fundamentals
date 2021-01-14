import matplotlib.pyplot as plt
def plot_images(X,y,num=4):
    print(X.shape,y.shape)
    _, axes = plt.subplots(nrows=1, ncols=num, figsize=(10, 3))
    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Label:{label}')