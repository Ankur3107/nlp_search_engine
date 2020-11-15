from full_search import final_search
from pprint import pprint

test = ['latest amir khan hindi 2019 movie',
        'salman khan horror movie',
        'salman khan horror movie 2019',
        'salu movie 2019',
        'latest salman khan hindi movie 2019',
        'latest saif ali khan in hindi 2019',
        'once upon time in mumbaai',
        'lagaan',
        'khan',
        'singh is king',
        'karina and salman khan movies',
        'hero no 1',
        'dil deevana',
        'shahrukh khn hindi movies',
        'shahrukh khn and salman kha hindi movies',
        'raj kumar movies',
        'shabana hindi movie',
        'raj kumar and salmna khan hindi horror movie 2019',
        'latest tamil drama movie',
        'arjun rampal movies 2018',
        'ajay devgan hindi horror movies prime vides',
        'hindi drama movie voot',
        '2019 movies',
        'punjabi 2019 movies',
        'khan 2019 movies',
        'doom 2',
        'gajanni',
        'salman khan don 2',
        'dil deevana samlam khan',
        'shahrukh khan don 2',
        'latest hindi action movie',
        'shahrukh khan latest movie',
        'shahrukh khan 2019 movie',
        'action 2019',
        'netflix tv movie']


for text in test:
  print("* Query :",text)
  pprint(final_search(text))