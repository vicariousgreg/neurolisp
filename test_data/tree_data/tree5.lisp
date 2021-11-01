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
  (setq tree6 '(x y z))

  (defun tree-sublis (subs tree)
      (let ((ret (tree-sublis-helper subs tree)))
          (if ret ret tree)))
  (defun tree-sublis-replace (subs tree)
      (if subs
        (if (expr-equal? (car (car subs)) tree)
          (cadr (car subs))
          (tree-sublis-replace (cdr subs) tree))
        NIL))
  (defun tree-sublis-helper (subs tree)
      (let ((replacement (tree-sublis-replace subs tree)))
        (cond
            (replacement replacement)
            ((atom tree) NIL)
            (true
              (let ((subtrees (forest-sublis-helper subs (cdr tree))))
                  (if subtrees
                      (cons (car tree) subtrees)
                      NIL))))))
  (defun forest-sublis-helper (subs subtrees)
      (if subtrees
          (let ((curr (tree-sublis-helper subs (car subtrees)))
                (rest (forest-sublis-helper subs (cdr subtrees))))
              (if (or curr rest)
                  (cons (if curr curr (car subtrees))
                      (if rest rest (cdr subtrees)))
                  NIL))
          NIL))

  (setq subs
      '((a (x y z)) ((b d e) y) (c z)))
  (test '(expr-equal? (tree-sublis subs tree1) '(x y z)) true)
  (test '(expr-equal? (tree-sublis subs tree5) '(a (f g) z y)) true)
  (test '(expr-equal? (tree-sublis subs tree6) tree6) true)

  'ALL_TESTS_PASSED
)
