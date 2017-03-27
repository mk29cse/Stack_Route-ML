
# coding: utf-8

# ## Numpy Exercises
# 
# ## Part - A
# 
# #### Make use of the following inbuilt api's
# 1. min
# 2. max
# 2. mean
# 3. argmin
# 4. argmax
# 5. linalg.solve
# 6. flatten
# 7. ravel
# 8. dot Product
# 9. \* for Matrix (Element wise multiplication)
# 10. arange followed by reshape
# 11. identity / eye
# 12. sum
# 13. diag
# 14. tril / triu
# 
# ## Part - B
# 
# #### For the all api's mentioned above implement your own version of it. Except linalg.solve. At minimum each implementation of yours should support 2d matrices.
# 

# ## Part -A

# In[2]:

import numpy as np


# In[ ]:

# Generating random 1D and 2D arrays to work on


# In[25]:

a = np.arange(0,10,2)
b = np.floor(10*np.random.random(10))
aM = np.floor(10*np.random.random(12)).reshape(3,4)
bM = np.ceil(10*np.random.random(12)).reshape(3,4)


# In[ ]:

# Finding min element's index across specified axis
# axis=1 means row wise and 0 means coloumns wise


# In[28]:

aM.min(axis=0)
aM.min(axis=1)


# In[29]:

bM


# In[ ]:

# Finding max element's index across specified axis
# axis=1 means row wise and 0 means coloumns wise


# In[31]:

bM.max(axis=0)
bM.max(axis=1)


# In[ ]:

# finding mean of all the element of matrix
# finding mean across specified axis
# axis = 1 , returns list with mean across each row
#axis = 0, returns list with mean across each column


# In[34]:

bM.mean()
bM.mean(axis=0)
bM.mean(axis=1)


# In[ ]:

#argmin returns the index of min element in the flatten up matrix, by default
# if axis specified, it returns a list with the index of the min element across specified axis


# In[39]:

np.argmin(bM)
np.argmin(bM, axis=0)
np.argmin(bM, axis=1)


# In[ ]:

# flatten returns a product of shape's element length list with all the matrix element


# In[178]:

bM.flatten()


# In[49]:

bM


# In[180]:

# flatten out the matrix
bM.ravel()


# In[53]:

bM.ravel().reshape(3,-1)

# reshape a given array/ matrix into m * n matrix
# In[55]:

np.dot(aM.reshape(3,4), bM.reshape(4,3))

# given 2 equal dimension matrix, returns a element wise product of provided matrix
# In[56]:

aM*bM


# In[58]:

np.arange(10).reshape(5,2)


# In[ ]:

#minimum of a ndarray


# In[ ]:

b.min()


# In[ ]:

#maximum of a ndarray


# In[8]:

b.max()


# In[ ]:

#mean of a ndarray


# In[9]:

b.mean()


# In[11]:

# return the indices of the minimum values along an axis
#np.argmin(array_like, axis,out=[array/optional])
np.argmin(b)


# In[12]:

np.argmax(b)


# In[18]:

#np.linalg.solve
#solve ax=b for x here a and b are matrix atleast 2D
m = np.array([[3,1], [1,2]])
n = np.array([9,8])
x = np.linalg.solve(m, n)
x


# In[19]:

# check if the dot product of m and x == n
np.allclose(np.dot(m,x),n)


# In[186]:

# generate an identity matrix of 3*3 
np.identity(3)


# In[62]:

np.identity(3, dtype=int)

#np.eye(num of rows, num of columns, k=0-main diagonal,+ve:upper diagonal,-ve lower diagonal, dtype)
# In[68]:

np.eye(5,5,k=-1)


# In[69]:

np.eye(5,5,k=2)


# In[72]:

np.eye(5,5,k=2).sum(axis=1)


# In[73]:

np.eye(5,5,k=2).sum(axis=0)


# In[77]:

t = np.eye(5,5,k=0)
t


# In[79]:

# if a 2D matrix is passed, it extracts the kth diagonal from the 2D matrix
np.diag(t,0)


# In[80]:

#if 1D is passed it creates a 2D matrix with kth diagonal element as the 1D matrix
np.diag([1,2,3,1])


# In[90]:

#takes an 2D matrix, return an array with the elements above kth diagonal as 0
a=np.floor(10*np.random.random(9)).reshape(3,3)
np.tril(a,0)


# In[92]:

np.triu(a,1)


# In[95]:

np.array([1,2,3]).shape


# In[185]:

flatten((np.array([[1,2,3,4],[2,3,4,5]])),'c')


# In[103]:

rows,cols = (np.array([[1,2,3,4],[2,3,4,5]])).shape
cols


# # Part -B

# In[104]:

def minn(array, axis=None ):
    if len(array.shape)==1:
        minn=99999
        for i in array:
            if i<minn:minn=i
        return minn
    elif axis == None:
        t=flatten(array)
        minn=99999
        for i in t:
            if i<minn:minn=i
        return minn
    
    elif axis == 0:
        l=[]
        rows,cols=array.shape
        
        for col in range(cols):
            minn=99999
            for row in range(rows):
                if array[row][col]<minn:minn=array[row][col]
            l.append(minn)
        return l
    
    elif axis == 1:
        l=[]
        rows,cols=array.shape
        
        for row in range(rows):
            minn=99999
            for col in range(cols):
                if array[row][col]<minn:minn=array[row][col]
            l.append(minn)
        return l


# In[ ]:

def ravel(array,order='r'):
    m = array.shape[0]
    n = array.shape[1]
    l=[]
    if order == 'r':
        for i in range(m):
            for j in range(n):
                l.append(array[i][j])
    elif order == 'c':
        for j in range(n):
            for i in range(m):
                l.append(array[i][j])
    return l


# In[132]:

def flatten(array,order='r'):
    m = array.shape[0]
    n = array.shape[1]
    l=[]
    if order == 'r':
        for i in range(m):
            for j in range(n):
                l.append(array[i][j])
    elif order == 'c':
        for j in range(n):
            for i in range(m):
                l.append(array[i][j])
    return l


# In[211]:

def summ(array, axis=None ):
    if len(array.shape)==1:
        summ=0
        for i in array:
            summ+=i
        return summ
    elif axis == None:
        t=flatten(array)
        summ=0
        for i in t:
            summ+=i
        return summ
    
    elif axis == 0:
        l=[]
        rows,cols=array.shape
        
        for col in range(cols):
            summ=0
            for row in range(rows):
                summ+=array[row][col]
            l.append(summ)
        return l
    
    elif axis == 1:
        l=[]
        rows,cols=array.shape
        
        for row in range(rows):
            summ=0
            for col in range(cols):
                summ+=array[row][col]
            l.append(summ)
        return l


# In[134]:

def meann(array, axis=None ):
    if len(array.shape)==1:
        summ=0
        for i in array:
            summ+=i
        return summ/len(array)
    elif axis == None:
        t=flatten(array)
        summ=0
        for i in t:
            summ+=i
        return summ/len(t)
    
    elif axis == 0:
        l=[]
        rows,cols=array.shape
        
        for col in range(cols):
            summ=0
            for row in range(rows):
                summ+=array[row][col]
            l.append(summ/rows)
        return l
    
    elif axis == 1:
        l=[]
        rows,cols=array.shape
        
        for row in range(rows):
            summ=0
            for col in range(cols):
                summ+=array[row][col]
            l.append(summ/cols)
        return l


# In[162]:

def argmin(array, axis=None ):
    if len(array.shape)==1:
        minn=99999
        index=-1
        for i in range(len(array)):
            if array[i]<minn:
                minn=array[i]
                index=i
        return index
    elif axis == None:
        t=flatten(array)
        minn=99999
        index=-1
        for i in range(len(t)):
            if t[i]<minn:
                minn=t[i]
                index=i
        return index
    
    elif axis == 0:
        l=[]
        rows,cols=array.shape
        
        for col in range(cols):
            minn=99999
            for row in range(rows):
                if array[row][col]<minn:
                    minn=array[row][col]
                    index=row
            l.append(index)
        return l
    
    elif axis == 1:
        l=[]
        rows,cols=array.shape
        
        for row in range(rows):
            minn=99999
            for col in range(cols):
                if array[row][col]<minn:
                    minn=array[row][col]
                    index=col
            l.append(index)
        return l


# In[166]:

def argmax(array, axis=None ):
    if len(array.shape)==1:
        maxx=-999
        for i in range(len(array)):
            if array[i]>maxx:
                maxx=array[i]
                index=i
        return index
    elif axis == None:
        t=flatten(array)
        maxx=-999
        for i in range(len(t)):
            if t[i]>maxx:
                maxx=t[i]
                index=i
        return index
    
    elif axis == 0:
        l=[]
        rows,cols=array.shape
        
        for col in range(cols):
            maxx=-999
            for row in range(rows):
                if array[row][col]>maxx:
                    maxx=array[row][col]
                    index=row
            l.append(index)
        return l
    
    elif axis == 1:
        l=[]
        rows,cols=array.shape
        
        for row in range(rows):
            maxx=-999
            for col in range(cols):
                if array[row][col]>maxx:
                    maxx=array[row][col]
                    index=col
            l.append(index)
        return l


# In[169]:

argmax(np.array([[2,6,0,4],[1,2,4,0]]))


# In[136]:

def maxx(array, axis=None ):
    if len(array.shape)==1:
        maxx=-9999
        for i in array:
            if i>maxx:maxx=i
        return maxx
    elif axis == None:
        t=flatten(array)
        maxx=-9999
        for i in t:
            if i>maxx:maxx=i
        return max
    
    elif axis == 0:
        l=[]
        rows,cols=array.shape
        
        for col in range(cols):
            maxx=-9999
            for row in range(rows):
                if array[row][col]<min:min=array[row][col]
            l.append(min)
        return l
    
    elif axis == 1:
        l=[]
        rows,cols=array.shape
        
        for row in range(rows):
            min=99999
            for col in range(cols):
                if array[row][col]<min:min=array[row][col]
            l.append(min)
        return l


# In[191]:

def identity(n):
    array=np.zeros([n,n])
    for row in range(n):
        for col in range(n):
            if row==col:array[row][col]=1
            else:array[row][col]=0
    return array      


# In[210]:

np.eye(3,3,k=0).sum()


# In[71]:

def diag(v, k=0):
    if len(v.shape)==1:
        i=0
        N=len(v)
        if k==0: array = np.zeros((N,N))
        else :  
            N=N+abs(k)
            array = np.zeros((N, N))
        for row in range(N):
            for col in range(N):
                if col-row==k:
                    array[row][col]=v[i]
                    i+=1
                else: array[row][col]=0
        return array
        
    elif len(v.shape)==2:
        rows,cols=v.shape
        l=[]
        for row in range(rows):
            for col in range(cols):
                if col-row==k:
                    l.append(v[row][col])
        return np.array(l)
    


# In[72]:

x = np.arange(1,10).reshape((3,3))
x


# In[73]:

t=diag(diag(x),1)
t


# In[39]:

diag(np.arange(9).reshape(3,3))
np.arange(9).reshape(3,3)


# In[199]:

def eye(N, M=None, k=0):
    if M==None:
        array=np.zeros([N,N])
        M=N
    else : array=np.zeros([N,M])
    for row in range(N):
        for col in range(M):
            if col-row==k:
                array[row][col]=1
            else:
                array[row][col]=0
    return array
            


# In[88]:

def elementWise_product(A, B):
    rows, cols = A.shape
    C = np.zeros((rows, rows))
    for row in range(rows):
        for col in range(cols):
            C[row][col] = A[row][col] * B[row][col]
    return C  


# In[91]:

A=np.arange(1,10).reshape(3,3)
B=np.arange(1,10).reshape(3,3)


# In[92]:

elementWise_product(A,B)


# In[139]:

def reshape(a, newshape):
    if len(newshape) == 2:
        nrows, ncols =newshape
        arr = np.zeros(newshape)
        l=flatten(a)
        i=0
        for row in range(nrows):
            for col in range(ncols):
                arr[row][col] = l[i]
                i+=1
    return arr


# In[140]:

reshape(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]),(6,2))


# In[158]:

def dotProduct(a,b):
    arows, acols = a.shape
    brows, bcols = b.shape
    result = np.zeros((arows, bcols)) 
    if acols == brows:
        for row in range(arows):
            for col in range(bcols):
                result[row][col] = sum(a[row,:]*b[:,col])
    return result       
    


# In[159]:

a=np.arange(1,7).reshape(3,2)
b= np.arange(1,7).reshape(2,3)
print(a.dot(b))
print(dotProduct(a,b))


# In[120]:

def tril(m, k=0):
    rows, cols = m.shape
    array = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            if col-row > k:
                array[row][col]=0
            else:
                array[row][col]=m[row][col]
    return array


# In[124]:

def triu(m, k=0):
    rows, cols = m.shape
    array = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            if col-row < k:
                array[row][col]=0
            else:
                array[row][col]=m[row][col]
    return array


# In[127]:

triu(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))


# In[128]:

np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

