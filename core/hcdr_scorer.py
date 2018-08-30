from abc import ABC, abstractmethod

class HCDRDataScorer(ABC):
    
    @abstractmethod
    def score(self):
        """ This method take current ids and return Previous application score
        """
        pass