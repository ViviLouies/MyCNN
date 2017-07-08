% Here's an implementation of the sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
