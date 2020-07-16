data Op = Add | Sub | Mul | Div -- deriving Show

instance Show Op where
    show Add = "+"
    show Sub = "-"
    show Mul = "*"
    show Div = "/"

valid :: Op -> Int -> Int -> Bool
valid Add _ _ = True
valid Sub x y = x > y
valid Mul _ _ = True
valid Div x y = x `mod` y == 0

apply :: Op -> Int -> Int -> Int
apply Add x y = x + y
apply Sub x y = x - y
apply Mul x y = x * y
apply Div x y = x `div` y

-- Define expressions
data Expr = Val Int | 
            App Op Expr Expr | 
            NullExpr --deriving Show

instance Show Expr where
  show (Val n)     = show n
  show NullExpr    = show 'x'
  show (App o l r) = brak l ++ show o ++ brak r
                   where
                     brak (Val n) = show n
                     brak e = "(" ++ show e ++")"

eval :: Expr -> [Int]
eval (Val n) = [n | n > 0]
eval (App o l r) = [apply o x y | x <- eval l, 
                                  y <- eval r, valid o x y]

e1 :: Expr
e1 = (App Add (App Mul (Val 4) (Val 5)) (Val 3))

e2 :: Expr
e2 = (App Sub (Val 5) (Val 6))

e3 :: Expr
e3 = (App Sub NullExpr (Val 6))

takeV :: Int -> [a] -> [[a]]
takeV n xs = [take i xs | i <- [n..(length xs -n)] ]
dropV :: Int -> [a] -> [[a]]
dropV n xs = [drop i xs | i <- [n..(length xs -n)] ]

split :: Int -> [a] -> [([a],[a])]
split n xs = zip (takeV n xs) (dropV n xs)

-- All placements of x in list xs 
perms2 :: a -> [a] -> [[a]]
perms2 x xs = map (\(ys,zs) -> ys++[x]++zs) (split 0 xs)

-- All possible permutations
perms :: [a] -> [[a]]
perms [] = [[]]
perms (x:xs) = concat (map (perms2 x) (perms xs) )

combine :: Expr -> Expr -> [Expr]
combine l r = [App o l r | o <- [Add, Sub, Mul, Div]]

exprs :: [Int] -> [Expr]
exprs [ ] = [ ]
exprs [n] | n == 0 = [NullExpr]
          | otherwise = [Val n ]
exprs ns = [e | (ls, rs) <- split 1 ns, l <- exprs ls,
                 r <- exprs rs, e <- combine l r ]

result :: [Int] -> [Expr]
result ns = [e | xs <- perms ns,
                 e <- exprs xs]

zipThree :: [Expr] -> [Expr] -> [Expr] -> [(Expr,Expr,Expr)]
zipThree [] _ _ = []
zipThree _ [] _ = []
zipThree _ _ [] = []
zipThree (x:xs) (y:ys) (z:zs) = (x,y,z) : zipThree xs ys zs

triple :: [Expr] -> [Expr] -> [Expr] -> [(Expr,Expr,Expr)]
triple xs ys zs = zipThree (reverse xs) (reverse ys) (reverse zs)

solution :: [Int] -> [Int] -> [Int] -> [String]
solution xs ys zs = [show z ++ " = " ++ show (head (eval y)) | (x,y,z) <- triple (result xs) (result ys) (result zs), eval x == eval y && eval x /= []]

-- solution :: [Int] -> [Int] -> [Int] -> [(Expr,Expr,Expr,Int)]
-- solution xs ys zs = [(x,y,z, head (eval y)) | (x,y,z) <- triple (result xs) (result ys) (result zs), eval x == eval y && eval x /= []]

-- solution :: [Int] -> [Int] -> [Int] -> [(Expr,Int)]
-- solution xs ys zs = [(z, head (eval y))| (x,y,z) <- zipThree (result xs) (result ys) (result zs), eval x == eval y && eval x /= []]

-- (a,b) = head $ solution [55,58,21] [73,83,14] [28,31,0]
-- string = show a
