/*
 * Created by Minseok Chu on 2023-12-26.
 */

#ifndef LINKEDLISTCONSOLE_NODE_H
#define LINKEDLISTCONSOLE_NODE_H

#include <iostream>

#include "linked_list.h"

namespace linked_list {

class SimpleList;

}  // namespace linked_list

struct MedicineInformation {
  std::string medicine_name;
  int medicine_id = -1;
  int medicine_amount;
};

struct PatientInformation {
  std::string name;

  linked_list::SimpleList* medicine_list;
};

struct Node {
  Node* previous;
  Node* next;
  int key;

  MedicineInformation medicine_info;
  PatientInformation patient_info;
};

#endif  // LINKEDLISTCONSOLE_NODE_H
