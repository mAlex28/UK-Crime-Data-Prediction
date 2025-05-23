# UK-Crime-Data-Prediction

Predicts UK Crime data based on Postcode.

All data used is publicly available from UK Police APIs under the Open Government Licence. No personally identifiable information is used or inferred.

Reverse geocode to get all the postcodes relating to a latitude and longitude - postcodes.io

Predictions are based on past 6 months.

All crime categories are recategorised as follows:
'violent-crime': 'violence',
'robbery': 'violence',
'shoplifting': 'theft',
'burglary': 'theft',
'vehicle-crime': 'theft',
'bicycle-theft': 'theft',
'other-theft': 'theft',
'anti-social-behaviour': 'anti-social',
'public-order': 'anti-social',
'criminal-damage-arson': 'anti-social',
'drugs': 'drugs',
'possession-of-weapons': 'other',
'other-crime': 'other'
