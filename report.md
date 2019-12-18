# OCR assignment report

## Feature Extraction (Max 200 Words)
Basically, the implementation idea is using PCA to get 10 features, 
which constructs features that best preserves the spread of the data.
But need to notice that, generating PCA only done during loading train
data. And during loading test page, the reduce dimensions method will 
use data stored in the model before.

## Classifier (Max 200 Words)
Two versions are included in the code, nearest neighbors classifier
and knn classifier, but only nn classifier is actually used. Actually,
knn classifier can perform very well on noisy page as follows:
- Page 1: score = 86.3% correct
- Page 2: score = 87.1% correct
- Page 3: score = 81.5% correct
- Page 4: score = 65.6% correct
- Page 5: score = 51.9% correct
- Page 6: score = 43.9% correct  

It is obvious that the score for noisy pages is improved, but that of
clean page does decrease. And it is confused to find how to determine
a page is noisy or not, then knn classifier not actually included. But
combining it should be an improvement for the accuracy.



## Error Correction (Max 200 Words)
Error correction has been attempted, but not actually been used too.
The idea is to extract word from labels, and compare that with words 
in the dictionary(stored in the model during training stage). Then 
trying to find a closest one to replace and storing labels included.
The reason why excluded is that the performance with error correction.
This method took extremely long time, and did not improve the score much.

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: score = 98.0% correct
- Page 2: score = 97.5% correct
- Page 3: score = 76.3% correct
- Page 4: score = 50.3% correct
- Page 5: score = 36.4% correct
- Page 6: score = 27.8% correct


## Other information (Optional, Max 100 words)

