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

  'ALL_TESTS_PASSED
)
