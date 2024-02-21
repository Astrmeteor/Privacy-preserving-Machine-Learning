import random
import numpy as np
import matplotlib.pyplot as plt


def select_probability(num_ele=10, epoch=2, append_method="direct"):

    # The number of element: num_ele
    # The iteration: epoch
    original_array = np.array(range(num_ele))
    print(f"Original element list: {original_array}")
    print(f"Append method: {append_method}")
    unique_probability = 100 * (1 / len(original_array))
    print(f"Probability of selecting each element: {unique_probability:.2f}%")
    # Count percentage of each element
    percentages = np.bincount(original_array) / original_array.size * 100
    for i, percentage in enumerate(percentages):
        print(f"Element {i} percentage: {percentage:.2f}%")
    print("--------------------------------------------------------------")

    # Generate temporary list for processing
    temp_array = original_array.copy()
    # Dictionary for record, epoch: probability
    dict_pro = {}

    for i in range(epoch):
        print(f"Iteration {i+1}:")
        print(f"Chosen array: {temp_array}")
        # Choose one element randomly
        chosen_element = np.random.choice(temp_array, size=1)
        # Add chose element into tem_arrau for next choosing
        # 1） append to the rear directly
        if append_method == "direct":
            temp_array = np.append(temp_array, chosen_element)
        # 2 )
        elif append_method == "insert":
            temp_array = np.insert(temp_array, np.where(temp_array == chosen_element)[0][-1] + 1, chosen_element)

        print(f"Chosen element: {chosen_element[0]}\nNew array: {temp_array}")

        # unique_elements = len(set(temp_array))
        unique_probability = 100 * (1 / len(temp_array))
        print(f"Probability of selecting each element after iteration {i+1}: {unique_probability:.2f}%")

        # Count percentage of each element
        percentages = np.bincount(temp_array) / temp_array.size * 100
        for j, percentage in enumerate(percentages):
            print(f"Element {j} percentage: {percentage:.2f}%")
        print("--------------------------------------------------------------")
        dict_pro[i] = percentages
    print(dict_pro)
    return dict_pro


def threed_plot(X, Y, Z):
    # x, y = np.meshgrid(X, Y)

    # Plot direct
    z1 = np.vstack([Z["direct"][i] for i in range(len(X))]).T
    print(z1)

    # Plot insert
    z2 = np.vstack([Z["insert"][i] for i in range(len(X))]).T

    # Figure
    fig = plt.figure(figsize=(20, 10))

    # Plot figure of direct
    ax = fig.add_subplot(111, projection='3d')
    for i in range(z1.shape[0]):
        ax.plot(X, np.tile(Y[i], z1.shape[1]), z1[i], color='blue', linestyle=':', label=f'Direct {i}')

    # Plot figure of insert
    for i in range(z2.shape[0]):
        ax.plot(X, np.tile(Y[i], z2.shape[1]), z2[i], color='red', linestyle='-', label=f'Insert {i}')

    # 设置图形的标签
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Element')
    ax.set_zlabel('Probability (%)')
    ax.set_box_aspect([1, 1, 2])
    # ax.legend()
    plt.legend(bbox_to_anchor=(1.3, 1), loc='upper left')

    ax.relim()
    ax.autoscale_view()
    ax.set_yticks(Y)
    ax.set_zticks(np.arange(0, 51, 5))
    plt.show()


if __name__ == "__main__":
    num_ele = 10
    epoch = 100
    append_method = ["direct", "insert"]
    # Dictionary, append_method: probability of each epoch
    dict_pro = {}
    for i in range(len(append_method)):
        pro = select_probability(num_ele=num_ele, epoch=epoch, append_method=append_method[i])
        dict_pro[append_method[i]] = pro
    print(dict_pro)
    x_axis = np.array(range(epoch))
    y_axis = np.array(range(num_ele))
    z_axis = dict_pro
    threed_plot(x_axis, y_axis, z_axis)
