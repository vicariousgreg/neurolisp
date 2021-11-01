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

  (defun copy-forest (subtrees)
      (if (not subtrees) NIL
          (cons
              (copy-tree (car subtrees))
              (copy-forest (cdr subtrees)))))
  (defun copy-tree (tree)
      (if (atom tree) tree
          (cons (car tree)
              (copy-forest (cdr tree)))))

  (test '(tree-equal? (copy-tree tree1) tree1) true)
  (test '(tree-equal? (copy-tree tree5) tree5) true)

  'ALL_TESTS_PASSED
)
