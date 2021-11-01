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

  (defun tree-equal? (x y)
      (if (or (atom x) (atom y))
          (eq x y)
          (and (eq (car x) (car y))
              (forest-equal? (cdr x) (cdr y)))))
  (defun forest-equal? (x y)
      (if (or (not x) (not y))
          (eq x y)
          (and (tree-equal? (car x) (car y))
              (forest-equal? (cdr x) (cdr y)))))

  (test '(tree-equal? tree3 tree4) false)
  (test '(tree-equal? tree5 tree5) true)
  (test '(tree-equal? tree1 tree2) false)
  (test '(tree-equal? tree5 tree2) false)

  'ALL_TESTS_PASSED
)
