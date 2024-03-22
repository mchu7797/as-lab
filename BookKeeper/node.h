/*
 * Created by Minseok Chu on 2023-12-26.
 */

#ifndef LINKEDLISTCONSOLE_NODE_H
#define LINKEDLISTCONSOLE_NODE_H

#include <iostream>

namespace linked_list {

struct Node {
 public:
  Node* previous;
  Node* next;
  int id;

  int left_entry;
  int right_entry;

  std::string left_entry_comment;
  std::string right_entry_comment;

  std::string transaction_comment;
};

}  // namespace linked_list

#endif  // LINKEDLISTCONSOLE_NODE_H
