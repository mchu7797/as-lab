/*
 * Created by Minseok Chu on 2023-12-27.
 */

#ifndef LINKEDLISTCONSOLE_LINKED_LIST_CONSOLE_H
#define LINKEDLISTCONSOLE_LINKED_LIST_CONSOLE_H

#include <iostream>

#include "linked_list.h"

namespace linked_list_console {

class LinkedListConsole {
 public:
  LinkedListConsole();
  void Start();

 private:
  errno_t ReadFromConsole(std::string const& string_to_print, int length,
                          int* results);
  void ClearConsole();
  void ShowMenu();
  void ShowElements();
  void ShowError();

  linked_list::SimpleList* linked_list_;
};

}  // namespace linked_list_console

#endif  // LINKEDLISTCONSOLE_LINKED_LIST_CONSOLE_H
