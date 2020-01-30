function loss = computeLoss(grad, grad_11, grad_22)
grad1_diff = sum(abs(grad - grad_11), 3);
grad2_diff = sum(abs(grad - grad_22), 3);
grad1_sum = max(sum(abs(grad_11), 3), 0.1);
grad2_sum = max(sum(abs(grad_22), 3), 0.1);

grad1_ratio = grad1_diff ./ grad1_sum;
grad2_ratio = grad2_diff ./ grad2_sum;

loss = min(grad1_ratio, grad2_ratio);

end