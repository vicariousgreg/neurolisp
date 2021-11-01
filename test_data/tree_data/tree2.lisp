(progn
  (defun test (expr target)
      (if (not (eq (eval expr) target))
        (error (list target 'NOT_EQUAL expr))))

  (setq tree1 'a)
  (setq tree5 '(a (f g) c (b d e)))

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

  'ALL_TESTS_PASSED
)
