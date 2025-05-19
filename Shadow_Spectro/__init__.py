
"""
Python Module to do shadow spectroscopy 
based on the following articles:  "Algorithmic Shadow Spectroscopy" from  "Hans Hon Sang Chan", "Richard Meister", "Matthew L. Goh", "Bálint Koczor"

"""

__version__="1.0.0"
__date__=""
__author__="Hugo PAGES"

__email__="hugo.pages@etu.unistra.fr"

    
       
__articles__ = [
    {
        "title": "Algorithmic Shadow Spectroscopy",
        "author": ["Hans Hon Sang Chan", "Richard Meister", "Matthew L. Goh", "Bálint Koczor"],
        "year": 2024,
        "eprint": "2212.11036",
        "archivePrefix": "arXiv",
        "primaryClass": "quant-ph",
        "url": "https://arxiv.org/abs/2212.11036"
    }
]

from .ClassicalShadow import ClassicalShadow
from .ShadowSpectro import ShadowSpectro
from .Spectroscopy import Spectroscopy