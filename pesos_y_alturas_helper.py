import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def eval_model(df_train, df_valid,binsA=10, binsP=10):
    hist_M=np.histogram2d(df_train[df_train["Genero"]=="Mujer"]["Altura"].values,
                       df_train[df_train["Genero"]=="Mujer"]["Peso"].values,
                       bins=[binsA,binsP],
                        range=[[df_train["Altura"].min(), df_train["Altura"].max()],
                               [df_train["Peso"].min(), df_train["Peso"].max()]])
    hist_H=np.histogram2d(df_train[df_train["Genero"]=="Hombre"]["Altura"].values,
                       df_train[df_train["Genero"]=="Hombre"]["Peso"].values,
                       bins=[binsA,binsP],
                       range=[[df_train["Altura"].min(), df_train["Altura"].max()],
                               [df_train["Peso"].min(), df_train["Peso"].max()]])
    idx_A=np.digitize(df_valid["Altura"].values,hist_M[1])-1
    idx_P=np.digitize(df_valid["Peso"].values,hist_M[2])-1
    idx_P[idx_P==-1]=0
    idx_P[idx_P==binsP]=binsP-1
    idx_A[idx_A==-1]=0
    idx_A[idx_A==binsA]=binsA-1
    pred=hist_M[0][list(idx_A),list(idx_P)] > hist_H[0][list(idx_A),list(idx_P)]
    pred= pred == (df_valid["Genero"]=="Mujer")
    return sum(pred)/len(pred)

def plot_model(df_train,df_valid,binsA=10,binsP=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist_M=np.histogram2d(df_train[df_train["Genero"]=="Mujer"]["Altura"].values,
                       df_train[df_train["Genero"]=="Mujer"]["Peso"].values,
                       bins=[binsA,binsP],
                        range=[[df_train["Altura"].min(), df_train["Altura"].max()],
                               [df_train["Peso"].min(), df_train["Peso"].max()]])
    hist_H=np.histogram2d(df_train[df_train["Genero"]=="Hombre"]["Altura"].values,
                       df_train[df_train["Genero"]=="Hombre"]["Peso"].values,
                       bins=[binsA,binsP],
                       range=[[df_train["Altura"].min(), df_train["Altura"].max()],
                               [df_train["Peso"].min(), df_train["Peso"].max()]])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(hist_M[1][:-1] + 0.25, hist_M[2][:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz_M = hist_M[0].ravel()
    dz_H = hist_H[0].ravel()

    ax.bar3d(xpos, ypos, zpos, hist_M[1][1]-hist_M[1][0], hist_M[2][1]-hist_M[2][0], dz_M, zsort='average',color='b',alpha=0.5)
    ax.bar3d(xpos, ypos, zpos, hist_H[1][1]-hist_H[1][0], hist_H[2][1]-hist_H[2][0], dz_H, zsort='average',color='r',alpha=0.5)
    
    plt.show()