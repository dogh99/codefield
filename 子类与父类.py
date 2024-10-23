class bio:
    def __init__(self,n,s):
        self.name=n
        self.sex=s
    def breathe(self):
        print(f"{self.name}在呼吸")

class human(bio):
    def __init__(self,n1,s2):
        super().__init__(n1,s2)
        self.tail=True

h1=human(1,2)
print(h1.name)
h1.breathe()