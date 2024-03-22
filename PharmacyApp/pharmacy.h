/*
 * Created by Minseok Chu on 2023-12-27.
 */

#ifndef PHARMACYAPP_PHARMACY_H
#define PHARMACYAPP_PHARMACY_H

#include <iostream>

#include "linked_list.h"

namespace pharmacy_app {

class Pharmacy {
 public:
  Pharmacy();
  void Start();

 private:
  void ClearConsole();
  void ShowMenu();
  void ShowError();

  void ShowMedicineInfo(int index);
  void ShowMedicineInfo(Node* node);
  void ShowPatientInfo(int index);
  void ShowPatientInfo(Node* node);
  void ShowAllMedicineInfo();
  void ShowAllPatientInfo();

  void PauseConsole(bool error_caused);

  linked_list::SimpleList* patient_infos_;
  linked_list::SimpleList* medicine_infos_;
};

}  // namespace pharmacy_app

#endif  // PHARMACYAPP_PHARMACY_H
