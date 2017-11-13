# datamunger
python package for handling nan's and outliers using k-Nearest Neighbors

- Input is a dataframe for a distinct time where columns are features
- Nan's can be converted using a Nearest Neighbors algorithm that uses the other features to build the geometry
- outliers can be converted to nan's based on a desired tolerance and then imputed using the same framework
