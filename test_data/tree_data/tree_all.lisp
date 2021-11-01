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

  (test '(expr-equal? 'a 'b) false)
  (test '(expr-equal? 'a 'a) true)
  (test '(expr-equal? '(a (b c)) '(a (b c))) true)
  (test '(expr-equal? '(a (b (c))) '(a (b c))) false)

  (setq tree1 'a)
  (setq tree2 '(a b))
  (setq tree3 '(a (b c)))
  (setq tree4 '(b d e))
  (setq tree5 '(a (f g) c (b d e)))
  (setq tree6 '(x y z))
  (setq nottree1 '(a))
  (setq nottree2 '(a (b c) (d) e))
  (setq nottree3 '((a b c) (d e)))

  (defun is-tree? (expr)
      (or (atom expr)
          (and (listp expr)
              (atom (car expr))
              (cdr expr)
              (is-forest? (cdr expr)))))
  (defun is-forest? (expr)
      (or (not expr)
          (and (is-tree? (car expr))
               (is-forest? (cdr expr)))))

  (test '(is-tree? tree1) true)
  (test '(is-tree? tree2) true)
  (test '(is-tree? tree3) true)
  (test '(is-tree? tree4) true)
  (test '(is-tree? tree5) true)
  (test '(is-tree? nottree1) false)
  (test '(is-tree? nottree2) false)
  (test '(is-tree? nottree3) false)

  (defun tree-contains? (elm tree)
      (cond
          ((atom tree) (eq elm tree))
          (true (or (eq (car tree) elm)
                 (forest-contains? elm (cdr tree))))))

  (defun forest-contains? (elm forest)
      (and forest
          (or (tree-contains? elm (car forest))
              (forest-contains? elm (cdr forest)))))

  (test '(tree-contains? 'a tree1) true)
  (test '(tree-contains? 'd tree5) true)
  (test '(tree-contains? 'h tree5) false)

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
