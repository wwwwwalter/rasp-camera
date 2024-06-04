#include <queue>  
#include <opencv2/opencv.hpp>  
  
template <typename T>  
class LimitedQueue {  
private:  
    std::queue<T> queue_;  
    size_t max_size_;  
    int pop_count = 0;
  
public:  
    LimitedQueue(size_t max_size) : max_size_(max_size) {}  
  
    void push(const T& item) {  
        if (queue_.size() >= max_size_) {  
            queue_.pop();  // 弹出队列头部的元素，以保持队列大小  
            std::cout << "pop" << ++pop_count<< std::endl;
        }  
        queue_.push(item);  
    }  
  
    T& front() {  
        return queue_.front();  
    }  
  
    const T& front() const {  
        return queue_.front();  
    }  
  
    void pop() {  
        queue_.pop();  
    }  
  
    bool empty() const {  
        return queue_.empty();  
    }  
  
    size_t size() const {  
        return queue_.size();  
    }  
  
    size_t max_size() const {  
        return max_size_;  
    }  
};  
  
