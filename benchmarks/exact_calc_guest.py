# Executed by real MicroPython inside the GPU RV32IM guest, not by host eval.
# Exact rational/decimal calc adapter for Qwenputer's public arithmetic protocol.
def gcd_int(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def nd(a):
    return len(str(abs(a)))

class R:
    def __init__(self, a, b=1):
        if not b:
            raise ValueError()
        if b < 0:
            a, b = -a, -b
        g = gcd_int(a, b)
        self.a, self.b = a//g, b//g
        if nd(self.a) > 62 or nd(self.b) > 62:
            raise ValueError()
    def __neg__(self):
        return R(-self.a, self.b)
    def __pos__(self):
        return self
    def __add__(self, y):
        a, b, c, d = self.a, self.b, y.a, y.b
        if (d!=1 and nd(a)+nd(d)>62) or (b!=1 and nd(c)+nd(b)>62) or (b!=1 and d!=1 and nd(b)+nd(d)>62):
            raise ValueError()
        return R(a*d+c*b, b*d)
    def __sub__(self, y):
        return self + (-y)
    def __mul__(self, y):
        if nd(self.a)+nd(y.a)>62 or (self.b!=1 and y.b!=1 and nd(self.b)+nd(y.b)>62):
            raise ValueError()
        return R(self.a*y.a, self.b*y.b)
    def __truediv__(self, y):
        if (y.b!=1 and nd(self.a)+nd(y.b)>62) or (self.b!=1 and nd(self.b)+nd(y.a)>62):
            raise ValueError()
        return R(self.a*y.b, self.b*y.a)
    def ints(self, y):
        if self.b!=1 or y.b!=1:
            raise ValueError()
        return self.a, y.a
    def __floordiv__(self, y):
        a,b=self.ints(y)
        return R(a//b)
    def __mod__(self, y):
        a,b=self.ints(y)
        return R(a%b)
    def __pow__(self, y):
        a,b=self.ints(y)
        if b<0 or b>999 or (nd(a)>1 and (nd(a)-1)*b>62):
            raise ValueError()
        return R(a**b)

def N(s):
    if '.' in s:
        a,b=s.split('.')
        return R(int((a or '0')+b), 10**len(b))
    return R(int(s))

def gcd(a,b):
    a,b=a.ints(b)
    return R(gcd_int(a,b))

def calculate(source):
    # Only the calculator's lexical grammar is admitted, before guest eval.
    rewritten=[]
    i=0
    decimal='.' in source
    if decimal and ('/' in source or '%' in source or '^' in source or '**' in source or 'gcd' in source):
        raise ValueError()
    while i<len(source):
        c=source[i]
        if '0'<=c<='9' or c=='.':
            j=i+1
            while j<len(source) and ('0'<=source[j]<='9' or source[j]=='.'):
                j+=1
            rewritten.append("N('"+source[i:j]+"')")
            i=j
        elif source[i:i+3]=='gcd':
            rewritten.append('gcd')
            i+=3
        elif c in '+-*/%(), \t\r\n':
            rewritten.append(c)
            i+=1
        elif c=='^':
            rewritten.append('**')
            i+=1
        else:
            raise ValueError()
    v=eval(''.join(rewritten))
    if not isinstance(v,R):
        raise ValueError()
    if not decimal:
        return str(v.a) if v.b==1 else str(v.a)+'/'+str(v.b)
    k=0
    scale=1
    while scale%v.b:
        scale*=10
        k+=1
        if k>256:
            raise ValueError()
    s=str(abs(v.a)*scale//v.b)
    if k:
        s='0'*(k+1-len(s))+s
        s=s[:-k]+'.'+s[-k:]
        s=s.rstrip('0').rstrip('.')
    if len(s.replace('.','').lstrip('0'))>62:
        raise ValueError()
    return ('-' if v.a<0 else '')+s
