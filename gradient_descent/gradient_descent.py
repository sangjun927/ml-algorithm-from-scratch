import numpy as np
import pandas as pd
from scipy import sparse

def proc_col(col):
    """Encodes a pandas column with values between 0 and n-1.
 
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)

def encode_data(df):
    """Encodes rating data with continous user and movie ids using 
    the helpful fast.ai function from above.
    
    Arguments:
      train_csv: a csv file with columns user_id,movie_id,rating 
    
    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
      
    """
    ### BEGIN SOLUTION
    user2idx, df['userId'], num_users = proc_col(df['userId'])
    movie2idx, df['movieId'], num_movies = proc_col(df['movieId'])
    ### END SOLUTION
    return df, num_users, num_movies

def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    ### BEGIN SOLUTION
    user2idx, _, _ = proc_col(df_train['userId'])
    movie2idx, _, _ = proc_col(df_train['movieId'])

    df_val['userId'] = df_val['userId'].map(user2idx)
    df_val['movieId'] = df_val['movieId'].map(movie2idx)

    df_val.dropna(inplace=True)    
    ### END SOLUTION
    return df_val

def create_embedings(n, K):
    """ Create a numpy random matrix of shape n, K
    
    The random matrix should be initialized with uniform values in (0, 6/K)
    Arguments:
    
    Inputs:
    n: number of items/users
    K: number of factors in the embeding 
    
    Returns:
    emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6*np.random.random((n, K)) / K
    return emb


def df2matrix(df, nrows, ncols, column_name="rating"):
    """ Returns a sparse matrix constructed from a dataframe
    
    This code assumes the df has columns: MovieID,UserID,Rating
    """
    values = df[column_name].values
    ind_movie = df['movieId'].values
    ind_user = df['userId'].values
    return sparse.csc_matrix((values,(ind_user, ind_movie)),shape=(nrows, ncols))

def sparse_multiply(df, emb_user, emb_movie):
    """ This function returns U*V^T element wise multi by R as a sparse matrix.
    
    It avoids creating the dense matrix U*V^T
    """
    
    df["Prediction"] = np.sum(emb_user[df["userId"].values]*emb_movie[df["movieId"].values], axis=1)
    return df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction")

def cost(df, emb_user, emb_movie):
    """ Computes mean square error
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      error(float): this is the MSE
    """
    ### BEGIN SOLUTION
    df["Prediction"] = np.sum(emb_user[df["userId"].values] * emb_movie[df["movieId"].values], axis=1)
    error = np.mean((df["rating"] - df["Prediction"]) ** 2)
    ### END SOLUTION
    return error

def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """ Computes finite difference on MSE(U, V).
    
    This function is used for testing the gradient function. 
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1)/e

def gradient(df, Y, emb_user, emb_movie):
    """ Computes the gradient.
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      d_emb_user
      d_emb_movie
    """
    ### BEGIN SOLUTION

    R = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    prediction = sparse_multiply(df, emb_user, emb_movie)
    delta = R - prediction
    grad_user = -2 * delta.dot(emb_movie) / len(df)
    grad_movie = -2 * delta.T.dot(emb_user) / len(df)
    return grad_user, grad_movie
    ### END SOLUTION
    return grad_user, grad_movie

# you can use a for loop to iterate through gradient descent
def gradient_descent(df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number of iterations.
    
    Prints training cost and validation cost (if df_val is not None) every 50 iterations.
    
    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    
    ### BEGIN SOLUTION
    momentum_u = np.zeros_like(emb_user)
    momentum_m = np.zeros_like(emb_movie)
    beta = 0.9  # Momentum term

    for i in range(iterations):
        grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)

        # Update momentum for user and movie embeddings
        momentum_u = beta * momentum_u + (1 - beta) * grad_user
        momentum_m = beta * momentum_m + (1 - beta) * grad_movie

        # Update embeddings
        emb_user -= learning_rate * momentum_u
        emb_movie -= learning_rate * momentum_m

        if i % 50 == 0 or i == iterations - 1:
            train_cost = cost(df, emb_user, emb_movie)
            print(f"Iteration {i}: Training cost = {train_cost:.4f}")
            if df_val is not None:
                val_cost = cost(df_val, emb_user, emb_movie)
                print(f"Iteration {i}: Validation cost = {val_cost:.4f}")
                
    ### END SOLUTION
    return emb_user, emb_movie


