(progn
  (defun test (expr target)
      (if (not (eq (eval expr) target))
        (error (list target 'NOT_EQUAL expr))))

  (defun expr-equal? (x y)
      (cond
          ((or (atom x) (atom y)) (eq x y))
          ((and (listp x) (listp y))
              (and (expr-equal? (car x) (car y))
                   (expr-equal? (cdr x) (cdr y))))
          (true false)))

  (setq tree1 'a)
  (setq tree3 '(a (b c)))
  (setq tree5 '(a (f g) c (b d e)))

  (defun tree-subst (new old tree)
      (let ((ret (tree-subst-helper new old tree)))
          (if ret ret tree)))
  (defun tree-subst-helper (new old tree)
      (cond
          ((expr-equal? tree old) new)
          ((atom tree) NIL)
          (true
            (let ((subtrees (forest-subst-helper new old (cdr tree))))
                (if subtrees
                    (cons (car tree) subtrees)
                    NIL)))))
  (defun forest-subst-helper (new old subtrees)
      (if subtrees
          (let ((curr (tree-subst-helper new old (car subtrees)))
                (rest (forest-subst-helper new old (cdr subtrees))))
              (if (or curr rest)
                  (cons (if curr curr (car subtrees))
                      (if rest rest (cdr subtrees)))
                  NIL))
          NIL))

  (test '(expr-equal? (tree-subst 'z 'a tree1) 'z) true)
  (test '(expr-equal? (tree-subst '(z a b) 'a tree1) '(z a b)) true)
  (test '(expr-equal? (tree-subst 'z '(b c) tree3) '(a z)) true)
  (test '(expr-equal? (tree-subst 'z 'g tree5) '(a (f z) c (b d e))) true)

  'ALL_TESTS_PASSED
)
