/*
 * Created by Minseok Chu on 2023-12-26.
 */

#include "linked_list.h"

#include <iostream>

#include "node.h"

namespace linked_list {

SimpleList::SimpleList() {
  this->head_ = nullptr;
  this->tail_ = nullptr;
  this->current_ = nullptr;
  this->count_ = 0;
}

SimpleList::~SimpleList() { DeleteAll(); }

Node* SimpleList::NewNode(int key) {
  this->current_ = new Node();

  this->current_->next = this->current_;
  this->current_->previous = this->current_;
  this->current_->key = key;

  return this->current_;
}

Node* SimpleList::Read(int index) {
  if (index < 0 || index > this->count_ - 1) {
    return nullptr;
  }

  if (this->head_ == nullptr || this->tail_ == nullptr) {
    return nullptr;
  }

  this->current_ = this->head_;

  while (index > 0) {
    this->current_ = this->current_->next;
    --index;
  }

  return this->current_;
}

void SimpleList::Traversal() {
  this->current_ = this->head_;

  for (int i = 0; i < this->count_; ++i) {
    std::cout << this->current_->key << " ";
    this->current_ = this->current_->next;
  }

  std::cout << std::endl;
}

Node* SimpleList::AppendFromHead(Node* new_node) {
  if (this->head_ == nullptr || this->tail_ == nullptr) {
    this->head_ = new_node;
    this->tail_ = new_node;
    ++this->count_;

    return new_node;
  }

  new_node->next = this->head_;
  this->head_->previous = new_node;
  this->head_ = new_node;
  this->current_ = new_node;

  ++this->count_;

  return this->current_;
}

Node* SimpleList::AppendFromTail(Node* new_node) {
  if (this->head_ == nullptr || this->tail_ == nullptr) {
    this->head_ = new_node;
    this->tail_ = new_node;
    ++this->count_;

    return new_node;
  }

  new_node->previous = this->tail_;
  this->tail_->next = new_node;
  this->tail_ = new_node;
  this->current_ = new_node;

  ++this->count_;

  return this->current_;
}

Node* SimpleList::InsertBefore(Node* new_node, int index) {
  if (this->count_ == 0) {
    return nullptr;
  }

  if (index < 0 || index > this->count_ - 1) {
    return nullptr;
  }

  this->current_ = this->head_;

  while (index > 0) {
    this->current_ = this->current_->next;
    --index;
  }

  new_node->next = this->current_;
  new_node->previous = this->current_->previous;

  this->current_->previous->next = new_node;
  this->current_->previous = new_node;

  ++this->count_;

  return new_node;
}

Node* SimpleList::InsertAfter(Node* new_node, int index) {
  if (this->count_ == 0) {
    return nullptr;
  }

  if (index < 0 || index > this->count_ - 1) {
    return nullptr;
  }

  this->current_ = this->head_;

  while (index > 0) {
    this->current_ = this->current_->next;
    --index;
  }

  new_node->previous = this->current_;
  new_node->next = this->current_->next;

  this->current_->next->previous = new_node;
  this->current_->next = new_node;

  ++this->count_;

  return this->current_;
}

Node* SimpleList::DeleteFromHead() {
  if (this->count_ == 0) {
    return nullptr;
  }

  this->current_ = this->head_;

  this->head_->next->previous = this->head_->next;
  this->head_ = this->head_->next;

  --this->count_;

  return this->current_;
}

Node* SimpleList::DeleteFromTail() {
  if (this->count_ == 0) {
    return nullptr;
  }

  this->current_ = this->head_;

  this->tail_->previous->next = this->tail_->previous;
  this->tail_ = this->tail_->previous;

  --this->count_;

  return this->current_;
}

Node* SimpleList::Delete(Node* node_to_delete) {
  if (this->count_ == 0 || node_to_delete == nullptr) {
    return nullptr;
  }

  this->current_ = this->head_;

  while (this->current_ != node_to_delete) {
    this->current_ = this->current_->next;
  }

  this->current_->next->previous = this->current_->previous;
  this->current_->previous->next = this->current_->next;

  --this->count_;

  return node_to_delete;
}

void SimpleList::DeleteAll() {
  this->current_ = this->head_;

  while (this->current_ != this->tail_) {
    this->current_ = this->current_->next;
    delete this->current_->previous;
  }

  delete this->current_;

  this->count_ = 0;
  this->head_ = nullptr;
  this->tail_ = nullptr;
  this->current_ = nullptr;
}

Node* SimpleList::Modify(Node* node_to_modify, int key) {
  if (this->count_ == 0 || node_to_modify == nullptr) {
    return nullptr;
  }

  this->current_ = this->head_;

  while (this->current_ != node_to_modify) {
    this->current_ = this->current_->next;
  }

  this->current_->key = key;

  return this->current_;
}

Node* SimpleList::LinearSearchByUnique(int key) {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    std::cout << "리스트가 비어 있습니다!";
    return nullptr;
  }

  this->current_ = this->head_;

  int i = 0;

  while (this->current_ != this->tail_) {
    if (this->current_->key == key) {
      break;
    }

    this->current_ = this->current_->next;
    ++i;
  }

  if (this->current_->key != key) {
    std::cout << "결과 :  찾지 못 함!";
    return nullptr;
  }

  std::cout << "결과 : " << i;

  return this->current_;
}

void SimpleList::LinearSearchByDuplicate(int key, int* const size,
                                         Node** results[]) {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    std::cout << "리스트가 비어 있습니다!";
    return;
  }

  *size = 0;
  *results = new Node*[this->count_];
  this->current_ = this->head_;

  std::cout << "결과 : ";

  for (int i = 0; i < this->count_; ++i) {
    if (this->Read(i)->key == key) {
      std::cout << i << " ";
      (*results)[*size] = this->Read(i);
    }
  }

  std::cout << std::endl;
}

Node* SimpleList::BinarySearchByUnique(int key) {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    std::cout << "리스트가 비어 있습니다!";
    return nullptr;
  }

  if (key < this->head_->key || key > this->tail_->key) {
    std::cout << "결과 : 찾지 못함";
    return nullptr;
  }

  int left, right, middle;

  left = 0;
  right = this->count_;

  while (left <= right) {
    middle = (left + right) / 2;

    std::cout << middle << " ";

    if (this->Read(middle)->key == key) {
      std::cout << "결과 : " << middle << std::endl;
      return this->Read(middle);
    }

    if (this->Read(middle)->key < key) {
      left = middle + 1;
    } else {
      right = middle - 1;
    }
  }

  return nullptr;
}

void SimpleList::BinarySearchByDuplicate(int key, int* size, Node** results[]) {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    std::cout << "리스트가 비어 있습니다!";
    *size = -1;
    return;
  }

  if (key < this->head_->key || key > this->tail_->key) {
    std::cout << "NONE ";
    *size = -1;
    return;
  }

  int left, right, middle, is_duplicated;

  *results = new Node*[this->count_];

  *size = 0;
  left = 0;
  right = this->count_;

  while (left < right) {
    middle = (left + right) / 2;

    std::cout << middle << " ";

    if (this->Read(middle)->key == key) {
      is_duplicated = 0;

      for (int i = 0; i < *size; ++i) {
        if ((*results)[i] == this->Read(middle)) {
          is_duplicated = 1;
        }
      }

      if (!is_duplicated) {
        (*results)[*size] = this->Read(middle);
        ++*size;
        std::cout << middle << " ";
      }
    }

    if (this->Read(middle)->key < key) {
      left = middle + 1;
    } else {
      right = middle - 1;
    }
  }

  middle = left;

  std::cout << std::endl << "결과 : ";

  while (left < this->count_ &&
         this->Read(middle)->key == this->Read(left)->key) {
    std::cout << left << " ";
    (*results)[*size] = this->Read(left);
    ++left;
    ++*size;
  }

  std::cout << std::endl;
}

void SimpleList::SortByBubble() {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    return;
  }

  int temp;

  for (int i = 0; i < this->count_ - 1; ++i) {
    for (int j = 0; j < this->count_ - 1 - i; ++j) {
      if (this->Read(j)->key >= this->Read(j + 1)->key) {
        temp = this->Read(j)->key;
        this->Modify(this->Read(j), this->Read(j + 1)->key);
        this->Modify(this->Read(j + 1), temp);
      }
    }
  }
}

void SimpleList::SortByInsertion() {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    return;
  }

  int temp;

  for (int i = 0; i < this->count_; ++i) {
    for (int j = i; j > 0; --j) {
      if (this->Read(j - 1)->key > this->Read(j)->key) {
        std::cout << j << std::endl;
        temp = this->Read(j - 1)->key;
        this->Modify(this->Read(j - 1), this->Read(j)->key);
        this->Modify(this->Read(j), temp);
      }
    }
  }
}

void SimpleList::SortBySelection() {
  if (this->head_ == nullptr || this->tail_ == nullptr || this->count_ == 0) {
    return;
  }

  int lowest, temp;

  for (int i = 0; i < this->count_; ++i) {
    lowest = i;

    for (int j = i + 1; j < this->count_; ++j) {
      if (this->Read(lowest)->key >= this->Read(j)->key) {
        lowest = j;
      }
    }

    if (lowest != i) {
      temp = this->Read(lowest)->key;
      this->Modify(this->Read(lowest), this->Read(i)->key);
      this->Modify(this->Read(i), temp);
    }
  }
}

bool SimpleList::CheckListSorted() {
  if (this->count_ == 0 || this->head_ == nullptr || this->tail_ == nullptr) {
    return false;
  }

  int current_key;

  this->current_ = this->head_;
  current_key = this->current_->key;

  for (int i = 0; i < this->count_; ++i) {
    if (current_key > this->current_->key) {
      return false;
    }

    current_key = this->current_->key;
    this->current_ = this->current_->next;
  }

  return true;
}

}  // namespace linked_list