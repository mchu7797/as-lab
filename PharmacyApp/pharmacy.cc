/*
 * Created by Minseok Chu on 2023-12-27.
 */

#include "pharmacy.h"

#include <windows.h>

#include <iostream>
#include <string>

#include "node.h"

namespace pharmacy_app {

Pharmacy::Pharmacy() {
  this->medicine_infos_ = new linked_list::SimpleList();
  this->patient_infos_ = new linked_list::SimpleList();
}

void Pharmacy::ClearConsole() {
#ifdef WIN32
  HANDLE h_console;
  CONSOLE_SCREEN_BUFFER_INFO cbsi;
  SMALL_RECT scroll_rect;
  COORD scroll_target;
  CHAR_INFO fill;

  h_console = GetStdHandle(STD_OUTPUT_HANDLE);

  if (!GetConsoleScreenBufferInfo(h_console, &cbsi)) {
    return;
  }

  scroll_rect.Left = 0;
  scroll_rect.Top = 0;
  scroll_rect.Right = cbsi.dwSize.X;
  scroll_rect.Bottom = cbsi.dwSize.Y;

  scroll_target.X = 0;
  scroll_target.Y = (SHORT)(0 - cbsi.dwSize.Y);

  fill.Char.UnicodeChar = TEXT(' ');
  fill.Attributes = cbsi.wAttributes;

  ScrollConsoleScreenBuffer(h_console, &scroll_rect, nullptr, scroll_target,
                            &fill);

  cbsi.dwCursorPosition.X = 0;
  cbsi.dwCursorPosition.Y = 0;

  SetConsoleCursorPosition(h_console, cbsi.dwCursorPosition);
#else
  std::cout << "\e[1;1H\e[2J";
#endif
}

void Pharmacy::PauseConsole(bool error_caused) {
  if (error_caused) {
    std::cout << "잘못 입력하셨습니다! 돌아가려면 아무 키나 눌러 주세요!"
              << std::endl;
  } else {
    std::cout << "돌아가려면 아무 키나 눌러 주세요!" << std::endl;
  }

  if (std::cin.fail()) {
    std::cin.clear();
  }

  std::cin.ignore(std::cin.rdbuf()->in_avail());
  std::cin.get();
}

void Pharmacy::ShowMenu() {
  std::cout << "== 약국 관리 프로그램 ==" << std::endl;
  std::cout << "메뉴 : " << std::endl;
  std::cout << "1.  약품 목록" << std::endl;
  std::cout << "2.  약품 보기" << std::endl;
  std::cout << "3.  약품 추가" << std::endl;
  std::cout << "4.  약품 수정" << std::endl;
  std::cout << "5.  약품 제거" << std::endl;
  std::cout << "6.  처방전 목록" << std::endl;
  std::cout << "7.  처방전 보기" << std::endl;
  std::cout << "8.  처방전 추가" << std::endl;
  std::cout << "9.  처방전 수정" << std::endl;
  std::cout << "10. 처방전 제거" << std::endl;
  std::cout << "11. 처방전 약 수령" << std::endl;
  std::cout << "12. 프로그램 종료" << std::endl;
}

void Pharmacy::ShowMedicineInfo(int index) {
  Node *node = this->medicine_infos_->Read(index);

  if (node == nullptr) {
    return;
  }

  std::cout << "[" << node->key << "]"
            << " " << node->medicine_info.medicine_name << " - "
            << node->medicine_info.medicine_amount << std::endl;
}

void Pharmacy::ShowMedicineInfo(Node *node) {
  if (node == nullptr) {
    return;
  }

  std::cout << "[" << node->key << "]"
            << " " << node->medicine_info.medicine_name << " - "
            << node->medicine_info.medicine_amount << std::endl;
}

void Pharmacy::ShowPatientInfo(int index) {
  Node *node = this->patient_infos_->Read(index);

  if (node == nullptr) {
    return;
  }

  linked_list::SimpleList *medicine_list = node->patient_info.medicine_list;

  std::cout << "[" << node->key << "]"
            << " " << node->patient_info.name << " : " << std::endl;

  for (int i = 0; i < medicine_list->GetCount(); ++i) {
    std::cout << "\t"
              << "[" << i << "]"
              << " " << medicine_list->Read(i)->medicine_info.medicine_name
              << ", " << medicine_list->Read(i)->medicine_info.medicine_amount
              << std::endl;
  }
}

void Pharmacy::ShowPatientInfo(Node *node) {
  if (node == nullptr) {
    return;
  }

  linked_list::SimpleList *medicine_list = node->patient_info.medicine_list;

  std::cout << "[" << node->key << "]"
            << " " << node->patient_info.name << " : " << std::endl;

  for (int i = 0; i < medicine_list->GetCount(); ++i) {
    std::cout << "\t"
              << "[" << i << "]"
              << " " << medicine_list->Read(i)->medicine_info.medicine_name
              << ", " << medicine_list->Read(i)->medicine_info.medicine_amount
              << std::endl;
  }
}

void Pharmacy::ShowAllMedicineInfo() {
  for (int i = 0; i < medicine_infos_->GetCount(); ++i) {
    this->ShowMedicineInfo(i);
  }
}

void Pharmacy::ShowAllPatientInfo() {
  for (int i = 0; i < patient_infos_->GetCount(); ++i) {
    this->ShowPatientInfo(i);
  }
}

void Pharmacy::Start() {
  linked_list::SimpleList *medicine_list;
  Node *temp_node;
  std::string confirm;

  int menu_input, data_input, medicine_id, patient_id;
  bool is_exit = false;

  while (!is_exit) {
    this->ClearConsole();
    this->ShowMenu();

    std::cout << "메뉴 입력 : ";
    std::cin >> menu_input;

    switch (menu_input) {
      case 1:
        this->ShowAllMedicineInfo();

        this->PauseConsole(false);

        break;
      case 2:
        std::cout << "== 약품 검색 ==" << std::endl;

        std::cout << "찾고 싶은 약품의 아이디를 입력해 주세요! : ";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        this->ShowMedicineInfo(
            this->medicine_infos_->LinearSearchByUnique(data_input));

        this->PauseConsole(false);

        break;
      case 3:
        temp_node = medicine_infos_->NewNode();

        std::cout << "== 약품 추가 ==" << std::endl;

        std::cout << "약품 이름을 입력해 주세요! : ";
        std::cin >> temp_node->medicine_info.medicine_name;

        std::cout << "해당 약품의 수량을 입력해 주세요 : ";
        std::cin >> temp_node->medicine_info.medicine_amount;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        medicine_infos_->AppendFromTail(temp_node);

        break;
      case 4:
        std::cout << "== 약품 수정 ==" << std::endl;

        std::cout << "수정하고 싶은 약품의 아이디를 입력해 주세요! : ";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        temp_node = this->medicine_infos_->LinearSearchByUnique(data_input);

        if (temp_node == nullptr) {
          std::cout << "해당 아이디를 사용하는 데이터를 찾을 수 없습니다.";
          this->PauseConsole(false);
          break;
        }

        std::cout << "수정하실 약품 이름을 입력해 주세요! : ";
        std::cin >> temp_node->medicine_info.medicine_name;

        std::cout << "수정하실 약품의 수량을 입력해 주세요 : ";
        std::cin >> temp_node->medicine_info.medicine_amount;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        this->medicine_infos_->Modify(temp_node, temp_node->key);

        break;
      case 5:
        std::cout << "== 약품 삭제 ==" << std::endl;

        std::cout << "지우고 싶은 약품의 아이디 : ";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        temp_node = this->medicine_infos_->LinearSearchByUnique(data_input);

        if (temp_node == nullptr) {
          std::cout << "해당 아이디를 사용하는 데이터를 찾을 수 없습니다.";
          this->PauseConsole(false);
          break;
        }

        this->medicine_infos_->Delete(temp_node);
        break;
      case 6:
        this->ShowAllPatientInfo();

        this->PauseConsole(false);

        break;
      case 7:
        std::cout << "== 환자 검색 ==" << std::endl;

        std::cout << "찾고 싶은 환자의 아이디를 입력해 주세요!";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        this->ShowPatientInfo(
            this->patient_infos_->LinearSearchByUnique(data_input));

        this->PauseConsole(false);

        break;
      case 8:
        std::cout << "== 환자 추가 ==" << std::endl;

        temp_node = this->patient_infos_->NewNode();

        std::cout << "환자 이름을 입력해주세요! : ";

        std::cin >> temp_node->patient_info.name;

        temp_node->patient_info.medicine_list = new linked_list::SimpleList();

        int medicine_id;

        while (true) {
          Node *new_medicine = temp_node->patient_info.medicine_list->NewNode();

          std::cout << "-1을 눌러 입력을 끝낼 수 있습니다." << std::endl;
          std::cout << "복용해야 하는 약 아이디 : ";
          std::cin >> medicine_id;

          if (std::cin.fail()) {
            this->PauseConsole(true);
            continue;
          }

          if (medicine_id == -1) {
            break;
          }

          Node *current_medicine =
              this->medicine_infos_->LinearSearchByUnique(medicine_id);

          if (current_medicine == nullptr) {
            std::cout << "약 이름을 찾을 수 없습니다!!" << std::endl;
            continue;
          }

          new_medicine->medicine_info.medicine_id = medicine_id;
          new_medicine->medicine_info.medicine_name =
              current_medicine->medicine_info.medicine_name;

          std::cout << "해당 약품의 복용 갯수 : ";
          std::cin >> new_medicine->medicine_info.medicine_amount;

          if (std::cin.fail()) {
            this->PauseConsole(true);
            continue;
          }

          temp_node->patient_info.medicine_list->AppendFromTail(new_medicine);
        }

        this->patient_infos_->AppendFromTail(temp_node);

        break;
      case 9:
        std::cout << "== 환자 수정 ==" << std::endl;

        std::cout << "수정하고 싶은 환자의 아이디 : ";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        temp_node = this->patient_infos_->LinearSearchByUnique(data_input);

        if (temp_node == nullptr) {
          std::cout << "해당 아이디를 사용하는 데이터를 찾을 수 없습니다.";
          this->PauseConsole(false);
          break;
        }

        std::cout << "수정을 시작하면 처음부터 다시 작성해야 합니다. "
                  << "시도하시겠습니까? [Y, N] : ";

        std::cin >> confirm;

        if (!(confirm == "Y" || confirm == "y")) {
          std::cout << "갱신이 취소되었습니다!" << std::endl;
          this->PauseConsole(false);
          break;
        }

        patient_id = temp_node->key;

        this->patient_infos_->Delete(temp_node);

        temp_node = this->patient_infos_->NewNode(patient_id);

        std::cout << "환자 이름을 입력해주세요! : ";

        std::cin >> temp_node->patient_info.name;

        while (true) {
          Node *new_medicine = temp_node->patient_info.medicine_list->NewNode();

          std::cout << "-1을 눌러 입력을 끝낼 수 있습니다." << std::endl;
          std::cout << "복용해야 하는 약 아이디 : ";
          std::cin >> medicine_id;

          if (std::cin.fail()) {
            this->PauseConsole(true);
            continue;
          }

          if (medicine_id == -1) {
            break;
          }

          Node *current_medicine =
              this->medicine_infos_->LinearSearchByUnique(medicine_id);

          if (current_medicine == nullptr) {
            std::cout << "약 이름을 찾을 수 없습니다!!" << std::endl;
            continue;
          }

          std::cout << "해당 약품의 복용 갯수 : ";
          std::cin >> new_medicine->medicine_info.medicine_amount;

          if (std::cin.fail()) {
            this->PauseConsole(true);
            continue;
          }

          temp_node->patient_info.medicine_list->AppendFromTail(new_medicine);
        }

        this->patient_infos_->AppendFromTail(temp_node);

        break;
      case 10:
        std::cout << "== 환자 제거 ==" << std::endl;

        std::cout << "지우고 싶은 환자의 아이디 : ";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
          break;
        }

        temp_node = this->medicine_infos_->LinearSearchByUnique(data_input);

        if (temp_node == nullptr) {
          std::cout << "해당 아이디를 사용하는 데이터를 찾을 수 없습니다.";
          this->PauseConsole(false);
          break;
        }

        if (temp_node->patient_info.medicine_list == nullptr) {
          this->patient_infos_->Delete(temp_node);
          break;
        }

        temp_node->patient_info.medicine_list->DeleteAll();
        this->patient_infos_->Delete(temp_node);
        break;
      case 11:
        std::cout << "== 약 수령 ==" << std::endl;

        std::cout << "처방전 아이디 : ";
        std::cin >> data_input;

        if (std::cin.fail()) {
          this->PauseConsole(true);
        }

        temp_node = this->patient_infos_->Read(data_input);

        if (temp_node == nullptr) {
          std::cout << "해당 아이디를 사용하는 데이터를 찾을 수 없습니다.";
          this->PauseConsole(false);
          break;
        }
        
        medicine_list = temp_node->patient_info.medicine_list;

        for (int i = 0; i < medicine_list->GetCount(); ++i) {
          Node *temp_medicine = this->medicine_infos_->LinearSearchByUnique(
              medicine_list->Read(i)->medicine_info.medicine_id);

          if (temp_medicine == nullptr) {
            std::cout << "해당하는 약을 찾을 수 없습니다!";
            break;
          }

          if (temp_medicine->medicine_info.medicine_amount -
                  medicine_list->Read(i)->medicine_info.medicine_amount <
              0) {
            std::cout << "약이 부족해 처방할 수 없습니다!";
            break;
          } else {
            temp_medicine->medicine_info.medicine_amount -=
                medicine_list->Read(i)->medicine_info.medicine_amount;
          }
        }

        std::cout << "처방 완료!" << std::endl;

        this->PauseConsole(false);

        break;
      case 12:
        is_exit = true;
        break;
      default:
        this->PauseConsole(true);
        break;
    }
  }
}

}  // namespace pharmacy_app