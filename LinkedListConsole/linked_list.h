/*
 * Created by minse on 2023-12-26.
 */

#ifndef LINKEDLISTCONSOLE_LINKED_LIST_H
#define LINKEDLISTCONSOLE_LINKED_LIST_H

#include "node.h"

namespace linked_list {

class SimpleList {
 public:
  SimpleList();

  Node* NewNode(int key);
  Node* Read(int index);
  void Traversal();

  Node* AppendFromHead(Node* new_node);
  Node* AppendFromTail(Node* new_node);

  Node* InsertBefore(Node* new_node, int index);
  Node* InsertAfter(Node* new_node, int index);

  Node* DeleteFromHead();
  Node* DeleteFromTail();
  Node* Delete(Node* node_to_delete);
  void DeleteAll();

  Node* Modify(Node* node_to_modify, int key);

  Node* LinearSearchByUnique(int key);
  void LinearSearchByDuplicate(int key, int* size, Node** results[]);
  Node* BinarySearchByUnique(int key);
  void BinarySearchByDuplicate(int key, int* size, Node** results[]);

  void SortByBubble();
  void SortByInsertion();
  void SortBySelection();

  bool CheckListSorted();

  int GetCount() { return count_; }

  ~SimpleList();

 private:
  Node* head_;
  Node* tail_;
  Node* current_;
  int count_;
};

}  // namespace linked_list

#endif  // LINKEDLISTCONSOLE_LINKED_LIST_H
