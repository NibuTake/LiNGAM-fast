import numpy as np
import itertools
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn import linear_model
from scipy.optimize import linear_sum_assignment
from graphviz import Digraph
import scipy.stats as stats
from tqdm import tqdm

"""
Input:
    X:
        shape (n_samples, n_variables)
        
    print_result:
        boolean value. if True, the result will be printed.
        ex.) x ---|strength|---> y
        x is the cause, y is the effect.
    
    --Params for ICA--
    use_sklearn:
        boolean value. Choose negentropy(sklearn's FastICA) or kurtosis(default).
    n_iter:
        numerical value. It is used for sklearn's FastICA(default=1000).
    max_iter:
        numerical value. It is used for ICA(default=1000).
        
    --Params for Regression-    
    reg_type:
        string value. Choose linear(default) or lasso.
    criterion:
        string value. Choose aic or bic(default).
    max_iter_ls:
        numerical value. It is used for Lasso Regression(default=1000).
        
    --Params for verification--    
    shapiro:
        boolean value. if True, the shapiro wilk test is done(default=False).  
    
    --Params for Algorithm--
    algorithm:
        string value. Choose normal or fast(default).
    
Output:
    the matrix of causal structure.
    x = Bx + e. B is return value
Model:
    X = BX + e
    X = Ae
    W = Xe
    z = Ve = VAe = W_ze
    X: variables
    B: Causality Matrix
    e: Exogenous variable
    z: whitening variable(check, np.cov(z) is identity matrix))
LiNGAM Estimation:
    STEP1:
        Centerize and Whitening (X) and get [z,V].
    STEP2:
        Use (z), Estimate [W_z] using kurtosis base FastICA(Independent Component Analysis).
        Å¶Note W_z will be estimated by each rows. Finally, Use Gram-Schmidt orthogonalization.
        Å¶FastICA's Estimation can't identify "The correct row's order" and "Scale"
    STEP3:
        Use (W_z, inv(V)), estimate [A,PDW].
        Å¶Note P is Permutation matrix, D is Diagonal matrix.
    STEP4:
        Use [PDW] and acyclicity, estimate [P,D].
        Å¶Note B=(I-W) and diag(B) is I because of Acyclisity.
    STEP5:
        Use [PDW,P,inv(D)], estimate [W_hat]
    STEP6:
        Prune and Permutate (W_hat) by causal order.[P_dot]
        Algorithm == normal:
            Minimize The upper triangular part of the matrix.
        Algorithm == fast:
            Sequentially set the smallest element with 0 until it becomes triangularizable.
    STEP7:
        Linear Regression or Lasso Regression by causal order and replace B's value with coef.
        And get B.
Please Note that we do not take responsibility or liability for this code.
"""

class LiNGAM():
    def __init__(self,epsilon=1e-25):
        self.epsilon      = epsilon

    def fit(self, X, use_sklearn=True, print_result=False, n_iter=1000, random_state=0, max_iter=1000,
            reg_type="liner", criterion="bic", max_iter_ls=1000, shapiro=False, algorithm="normal"):
        self.criterion = criterion
        self.max_iter = max_iter
        self.max_iter_ls = max_iter_ls
        self.random_state = random_state
        self.print_result = print_result
        self.n_iter       = n_iter
        self.n_samples, self.n_dim  = X.shape
        self.shapiro = shapiro
        self.reg_type = reg_type
        self.algorithm = algorithm
        X_np = self._pd2np(X)
        
        self.X_center           = self._centerize(X_np)
        self.PDW                = self._calc_PDW(use_sklearn=use_sklearn)
        self.P_hat              = self._P_hat()
        self.D_hat,self.DW      = self._PW()
        self.B_hat              = self._B_hat()
        
        # Branching by algorithm
        if self.algorithm == "normal" :
            self.P_dot          = self._P_dot()
            self.B_prune        = self._B_prune()
        elif self.algorithm == "fast" :
            results = self._fast_calc_B_hat_P_dot(self.B_hat)
            self.P_dot          = results["P_dot"]
            self.B_prune        = results["B_hat"]
        else :
            return print("An avairable algorithm.")
        
        self.B                  = self._regression_B(X_np)
        return self.B

    def b_hat_(self) :
        return self.B_hat
    
    #if X is pandas DataFrame, convert numpy
    def _pd2np(self,X):
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(self.n_dim)]
        return X_np

    #centerize X by X's col
    def _centerize(self,X):
        return X - np.mean(X,axis=0)

    #whitening using Eigenvalue decomposition
    def _whitening(self,X):
        E, D, E_t = np.linalg.svd(np.cov(X, rowvar=0, bias=0), full_matrices=True)
        ##ïœÇ¶Ç»Ç´Ç·Ç¢ÇØÇ»Ç¢
        D = np.diag(D**(-1/2))
        V = E.dot(D).dot(E_t) #whitening matrix
        return V.dot(X.T),V
    """
    #whitening using Eigenvalue decomposition
    def _old_whitening(self,X):
        eigen, E = np.linalg.eig(np.cov(X, rowvar=0, bias=0))
        #eigen
        eigen[eigen<0] = -eigen[eigen<0]
        D = np.diag(eigen**(-1/2))
        V = E.dot(D).dot(E.T) #whitening matrix
        return V.dot(X.T),V
    """

    #Estimate W of Wz = s
    def _ICA(self,z,max_iter):
        np.random.seed(self.random_state)
        W_init = np.random.uniform(size=[self.n_dim,self.n_dim])
        W = np.zeros(W_init.shape)
        for i in range(self.n_dim):
            W[i,:] = self._calc_w(W_init[i,:], W, z, max_iter, i)
        return W

    #Estimate PDW
    def _PDW(self,W,V):
        A_tilde = np.linalg.inv(W)
        A = np.linalg.inv(V).dot(A_tilde)
        PDW = np.linalg.inv(A)
        return PDW

    #Estimate P
    def _P_hat(self):
        self.PDW[self.PDW == 0] = self.epsilon
        row_ind, col_ind = linear_sum_assignment(1/np.abs(self.PDW))
        P = np.zeros((len(row_ind),len(col_ind)))
        for i,j in  zip(row_ind,col_ind):
            P[i,j] = 1
        return P

    #Estimate D and DW
    def _PW(self):
        DW = self.P_hat.dot(self.PDW)
        return np.diag(np.diag(DW)),DW

    #Estimate W and B
    def _B_hat(self):
        W_hat = np.linalg.inv(self.D_hat).dot(self.DW)
        B_hat = np.eye(len(W_hat))-W_hat
        return B_hat

    # Functions added to reduce memory consumption
    # Generate substitution matrix by using matrix_key.
    def _generate_SubMatrix(self, matrix_key):
        seq = np.arange(1, self.n_dim + 1)
        SubMatrix = np.zeros((self.n_dim, self.n_dim), dtype=int)
        for i in seq :
                SubMatrix[i-1, matrix_key[i-1]-1] = 1
        return SubMatrix
    
    #Estimate P (permute B by causal order)
    # This function required a lot of memory, so we redefine it by using _generate_SubMatrix.
    def _P_dot(self):
        renew_score_flag = True
        score = []
        dim = self.n_dim
        seq = np.arange(1, dim + 1)
        for matrix_key in tqdm( list(itertools.permutations(seq)) ) :
            score_ = self._calc_PBP_upper(self._generate_SubMatrix(matrix_key), self.B_hat)
            if renew_score_flag == True :
                temp_score = score_
                temp_matrix_key = matrix_key
                renew_score_flag = False
            else :
                if temp_score > score_ :
                    temp_score = score_
                    temp_matrix_key = matrix_key
        return self._generate_SubMatrix(temp_matrix_key)

    #Prune B
    def _B_prune(self):
        B_prune = self.P_dot.dot(self.B_hat).dot(self.P_dot.T)
        for i in range(self.n_dim):
            for j in range(i,self.n_dim):
                B_prune[i,j] = 0
        return self.P_dot.T.dot(B_prune).dot(self.P_dot)

    #Replace B values with Regression coef
    # We add the LassoLarsIC as the selected model and ShapiroWilkTest for each residuals.
    def _regression_B(self,X):
        residual_flag_ = []
        causal_matrix = self.B_prune.copy()
        reg_list = {i:causal_matrix[i,:] != 0 for i in range(self.n_dim)}
        for i in range(self.n_dim):
            if np.sum(reg_list[i]) != 0:
                y_reg = X[:,i]
                X_reg = X.T[reg_list[i]].T
                if self.reg_type == "lasso" :
                    clf = linear_model.LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter_ls, precompute="auto")
                else :
                    clf = linear_model.LinearRegression()
                clf.fit(y=y_reg.reshape(self.n_samples,-1), X=X_reg.reshape(self.n_samples,-1))
                residual = y_reg.reshape(self.n_samples,-1) - clf.predict(X_reg.reshape(self.n_samples,-1))
                if self.shapiro == True :
                    if stats.shapiro(residual)[1] > 0.05 :
                        norm_flag = True
                    else :
                        norm_flag = False
                    residual_flag_.append(norm_flag)
                causal_matrix[i,reg_list[i]] = clf.coef_
            else:
                norm_flag = False
                residual_flag_.append(norm_flag)
        self.residual_flag = residual_flag_
        return causal_matrix

    #FastICA updates
    def _ICA_update(self,w,z):
        w = z.dot((w.T.dot(z)**3)) - 3*w
        w = w/np.sqrt(np.dot(w,w))
        return w

    #calculate w
    def _calc_w(self,w_init,W,z,max_iter,i):
        w_t_1  = w_init.copy()
        W_copy = W.copy()
        for iter_time in range(max_iter):
            w_t = self._ICA_update(w_t_1,z)
            #w_list.append(np.abs(np.dot(w_t,w_t_1)-1))
            if (np.abs(np.dot(w_t,w_t_1)-1) < self.epsilon) or (iter_time == (max_iter-1)):
                #without orthogonalization
                if i==0:
                    return w_t
                #orthogonalization
                else:
                    W_copy[i,:] = w_t
                    w_t = self._calc_gs(W=W_copy,i=i)
                    if (np.abs(np.dot(w_t,w_t_1)-1) < self.epsilon) or (iter_time == (max_iter-1)):
                        return w_t
                    else:
                        w_t_1 = w_t
            else:
                w_t_1 = w_t

    #Estimate W using Sklearn FastICA
    def _W_sklearn(self,X):
        A = FastICA(n_components=self.n_dim).fit(X).mixing_
        return np.linalg.inv(A)

    def _calc_PDW(self,use_sklearn):
        #use FastICA(kurtosis)
        if not use_sklearn:
            z, V   = self._whitening(self.X_center)
            W_z    = self._ICA(z,self.n_iter)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            PDW    = self._PDW(W_z, V)
        #use sklearn's FastICA(neg entropy)
        else:
            PDW    = self._W_sklearn(self.X_center)
        return PDW

    #GS orthogonalization
    def _calc_gs(self,W,i):
        w_i = W[i,:]
        w_add = np.zeros(w_i.shape)
        for j in range(i):
            w_j = W[j:(j+1),:].ravel()
            w_add = w_add + np.dot(w_i,w_j)*w_j
        w_i = w_i - w_add/i
        return w_i/np.sqrt(np.dot(w_i,w_i))

    #get sum of upper triangle value
    def _get_upper_triangle(self,mat):
        return np.diag(mat.dot(np.tri(self.n_dim))).sum()

    #P_dot
    def _get_P_dot_lists(self):
        base_array  = np.eye(N=1,M=self.n_dim).ravel().astype("int")
        base_array  = set(itertools.permutations(base_array))
        return np.array(list(itertools.permutations(base_array)))

    #get PBP to minimize upper triangle value
    def _calc_PBP_upper(self,P_dot,B_hat):
        return self._get_upper_triangle( P_dot.dot(B_hat).dot(P_dot.T)**2)

    #---------------------------------------------------------------------
    # Fast estimate B and P.    2018/05/12
    #---------------------------------------------------------------------
    
    # Decompose a matrix.
    # The matrix_ravel is the label.
    # The matrix_key is the value of matrix_ravel.
    def _pre_matrix(self, x):
        test_ravel = x.ravel()
        test_key = np.array(np.arange(len(x.ravel())))
        return {"matrix": test_ravel, "key": test_key}

    # Set the smallest elements of B to zero for the number.
    def _set_element_to_zero(self, matrix_ravel, matrix_key, num) :
        if num == 0:
            return {"matrix": matrix_ravel,"key": matrix_key}
        else:
            # Take the absolute value of the value.
            abs_matrix_ravel = abs(matrix_ravel)
            drop_key = abs_matrix_ravel.argmin()
            matrix_ravel = np.delete(matrix_ravel, drop_key)
            matrix_key = np.delete(matrix_key, drop_key)
            # Recursively call oneself.
            return self._set_element_to_zero(matrix_ravel, matrix_key, num-1)

    # Regenerate matrix by using matrix_ravel and matrix_key.
    def _regenerate_matrix(self, core_matrix, dim):
        zero_matrix = np.zeros((dim, dim) , dtype=float).ravel()
        for i, k in enumerate(core_matrix["key"]) :
            zero_matrix[k] = core_matrix["matrix"][i]
        return zero_matrix.reshape(dim,dim) 
    
    # Judge whether triangulation is possible or not.
    # If triangulation is possible, it returns the order, otherwise it returns a False as Boolean value.
    def _check_triangle(self, matrix):
        dim = len(matrix)
        break_flag = False
        for k in range(dim) :
            if (matrix[k].sum() == 0) or (len(matrix)==1):
                break_flag = True
                break_num = k
                break
        if break_flag == True:
            return break_num
        else :
            return False
    
    # Reduce dimention of matrix for break_num_list.
    def _dim_reduced_matrix(self, matrix, break_num_list):
        droped_matrix = matrix
        for causal_order in break_num_list:
            droped_matrix= np.delete(np.delete(droped_matrix, causal_order, axis=0), causal_order, axis=1)
        return droped_matrix

    # Set the n(n+1)/2 smallest in absolute value elements of B to zero.
    def first_droped_causal_order_matrix(self, matrix) :
        dim = len(matrix)
        core_matrix = self._set_element_to_zero(self._pre_matrix(matrix)["matrix"], self._pre_matrix(matrix)["key"], int(dim*(dim+1)/2))
        reg_matrix = self._regenerate_matrix(core_matrix, dim)
        causal_order = self._check_triangle(reg_matrix)
        return {"causal_order":causal_order, "matrix": reg_matrix, "core_matrix": core_matrix}

    # Run it recursively until it becomes triangularizable.
    def _droped_causal_order_matrix(self, core_matrix, dim, break_num_list=[]) :
        matrix = self._regenerate_matrix(core_matrix, dim)
        droped_matrix = self._dim_reduced_matrix(matrix, break_num_list)
        # When the dimension of the dimention reduced matrix becomes 1, it ends.
        if len(droped_matrix) == 1:
            return {"causal_order": break_num_list, "matrix": matrix}
        # Sequentially set the smallest element with 0 and verify triangulation possibility.
        while type(self._check_triangle(droped_matrix)) == bool:
            core_matrix = self._set_element_to_zero(core_matrix["matrix"], core_matrix["key"], 1)
            matrix = self._regenerate_matrix(core_matrix, dim)
            droped_matrix = self._dim_reduced_matrix(matrix, break_num_list)
        causal_order = self._check_triangle(droped_matrix)
        break_num_list.append(causal_order)
        return self._droped_causal_order_matrix(core_matrix, dim, break_num_list)

    # Because the reduced rows(break_num_list) are out of order, we reconstruct them as causal order.
    def _reconstruct_causal_order(self, causal_order):
        dim = self.n_dim
        causal_order_list = []
        dim_list = np.arange(dim)
        for k in np.arange(len(causal_order)) :
            causal_order_list.append(dim_list[causal_order[k]])
            dim_list = np.delete(dim_list, causal_order[k])
        return causal_order_list

    # Construct P_dot from causal order.
    def _construct_P_dot(self, cause):
        dim = self.n_dim
        P_dot = np.zeros((dim, dim))
        for k in np.arange(dim) :
            if not(k in cause) :
                cause.append(k)
        for i, j in enumerate(cause):
            P_dot[i, j] = 1
        return P_dot
    
    # Calculate B and P using the above all.
    def _fast_calc_B_hat_P_dot(self, x):
        result1 = self.first_droped_causal_order_matrix(x) 
        result2 = self._droped_causal_order_matrix(result1["core_matrix"], 
                                                  dim=len(result1["matrix"]), 
                                                  break_num_list=[result1["causal_order"]])
        causal_order = self._reconstruct_causal_order(result2["causal_order"])
        return {"B_hat": result2["matrix"],
                "P_dot": self._construct_P_dot(causal_order),
                "causal_order": causal_order}

    #print result
    def _result_print(self):
        if self.print_result:           
            for i,b in enumerate(self.columns):
                for j,a in enumerate(self.columns):
                    if self.B[i,j]!=0:
                        print(a,"---|%.3f|--->"%(self.B[i,j]),b)

    # Visualize cause of X by using Digraph.
    def visualize(self):
        B = self.B
        residual_flag = self.residual_flag
        columns = self.columns

        G = Digraph(format="png")
        for l, cl in enumerate(columns) :
            if len(residual_flag) > 0 :
                if residual_flag[l] == True :
                    color = "red"
                else :
                    color = "black"
            else :
                color = "black"
            non_edge_flg = 1
            for k, ck in enumerate(columns) :
                if B[l, k] != 0 :
                    G.edge(ck, cl, color=color, label= "    " + str(B[l, k].round(3)))
                    non_edge_flg = 0
            if non_edge_flg == 1:
                G.node(cl, shape="circle", color="blue")
        return G