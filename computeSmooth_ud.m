function [lr_12, lr_21, ud_12, ud_21] = computeSmooth_ud(im1, im2)

ud_padding = zeros(1, size(im1, 2), 3);
grad_ud_11 = im1(1:end-1, :, :) - im1(2:end, :, :);
grad_ud_11 = cat(1, ud_padding, grad_ud_11);

grad_ud_22 = im2(1:end-1, :, :) - im2(2:end, :, :);
grad_ud_22 = cat(1, grad_ud_22, ud_padding);

grad_ud_12 = im1(1:end-2, :, :) - im2(3:end, :, :);
grad_ud_12 = cat(1, ud_padding, grad_ud_12, ud_padding);


grad_ud_21 = im2 - im1;

grad_lr_11 = im1(1:end-1, 1:end-1, :) - im1(1:end-1, 2:end, :);
grad_lr_22 = im2(2:end, 1:end-1, :) - im2(2:end, 2:end, :);
grad_lr_12 = im1(1:end-1, 1:end-1, :) - im2(2:end, 2:end, :);
grad_lr_21 = im2(2:end, 1:end-1, :) - im1(1:end-1, 2:end, :);

lr_12 = computeLoss(grad_lr_12, grad_lr_11, grad_lr_22);
lr_21 = computeLoss(grad_lr_21, grad_lr_11, grad_lr_22);
ud_12 = computeLoss(grad_ud_12, grad_ud_11, grad_ud_22);
ud_21 = computeLoss(grad_ud_21, grad_ud_11, grad_ud_22);

end




