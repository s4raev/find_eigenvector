# Install

```bash
git clone https://github.com/s4raev/find_eigenvector.git
pip install -r requirements.txt
```
# Usage
```python
from find_eigenvectors.eigv import get_eig, test_matrix

A = np.array([
    [3, -2],
    [-4, 1]
], dtype=float)
values, vectors = get_eig(A)

test_matrix(A)
```
running built-in tests (matrix from the classbook)

```bash
python ./find_eigenvectors/eigv.py
```
# Function description
## get_eig(A)

Parameters: A 
> Matrix for which the eigenvalues and right eigenvectors will be computed

Returns: w (array), v (array)
> w - the eigenvalues, each repeated according to its multiplicity. The eigenvalues are not necessarily ordered. 

> v - the normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].