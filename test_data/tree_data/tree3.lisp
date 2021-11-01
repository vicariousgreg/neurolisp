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
  (setq tree2 '(a b))
  (setq tree3 '(a (b c)))
  (setq tree4 '(b d e))
  (setq tree5 '(a (f g) c (b d e)))

  (defun tree-prefix (tree)
      (tree-prefix-helper tree NIL))
  (defun tree-prefix-helper (tree seq)
      (if (atom tree)
          (cons tree seq)
          (cons (car tree) (forest-prefix-helper (cdr tree) seq))))
  (defun forest-prefix-helper (subtrees seq)
      (if subtrees
          (tree-prefix-helper
              (car subtrees)
              (forest-prefix-helper (cdr subtrees) seq))
          seq))

  (test '(expr-equal? (tree-prefix tree1) '(a)) true)
  (test '(expr-equal? (tree-prefix tree2) '(a b)) true)
  (test '(expr-equal? (tree-prefix tree3) '(a b c)) true)
  (test '(expr-equal? (tree-prefix tree4) '(b d e)) true)
  (test '(expr-equal? (tree-prefix tree5) '(a f g c b d e)) true)

  'ALL_TESTS_PASSED
)
