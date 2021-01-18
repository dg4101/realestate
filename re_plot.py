import matplotlib.pyplot as plt

def plot_prediction_plus_importance(y_test,y_pred, fn, perm_importance_means, algorism,now):
    fig = plt.figure(dpi=400)
    fig.suptitle("Boston House Prices Prediction by " + algorism)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.rcParams["font.size"] = 6

    plt.subplot(121)
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.scatter(y_pred, y_test, alpha=0.5) #訓練データの散布図
    plt.plot(y_pred, y_pred, c="r") #回帰直線
    plt.xlabel("Predicted Price")
    plt.ylabel("Actual prices")
    plt.xlim(10,40)
    plt.ylim(10,40)
    plt.tight_layout()

    plt.subplot(122)
    plt.barh(fn, perm_importance_means)
    plt.xlabel("Permutation Importance")
    plt.tight_layout()

    plt.savefig('images/boston-' + now + '-' + algorism + '.png')
    plt.show()