/*
 * Created by Minseok Chu on 2023-12-27.
 */

#include "linked_list_console.h"

#include <iostream>

#ifdef WIN32
#include <windows.h>

#include <string>
#endif

namespace linked_list_console {

LinkedListConsole::LinkedListConsole() {
  this->linked_list_ = new linked_list::SimpleList();
}

errno_t LinkedListConsole::ReadFromConsole(std::string const& string_to_print,
                                           int length, int* results) {
  if (!string_to_print.empty()) {
    std::cout << string_to_print;
  }

  std::cin.ignore(std::cin.rdbuf()->in_avail());

  std::string input;

  for (int i = 0; i < length; ++i) {
    std::cin >> input;

    for (char j : input) {
      if (!std::isdigit(j)) {
        return 2;
      }
    }

    try {
      results[i] = std::stoi(input);
    } catch (std::out_of_range const& e) {
      return 2;
    }
  }

  return 0;
}

void LinkedListConsole::ClearConsole() {
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

void LinkedListConsole::ShowElements() {
  if (linked_list_->GetCount() == 0) {
    return;
  }

  std::cout << "��� ���� : ";
  linked_list_->Traversal();
}

void LinkedListConsole::ShowError() {
  std::cout << "�߸� �Է��ϼ̽��ϴ�! ���͸� ������ �ٽ� �����մϴ�!";
  std::cin.ignore();
  std::cin.get();
}

void LinkedListConsole::ShowMenu() {
  std::cout << "�޴� : " << std::endl;
  std::cout << "1.  ��� �߰� (���)" << std::endl;
  std::cout << "2.  ��� �߰� (����)" << std::endl;
  std::cout << "3.  ��� ���� (��)" << std::endl;
  std::cout << "4.  ��� ���� (��)" << std::endl;
  std::cout << "5.  ��� ���� (���)" << std::endl;
  std::cout << "6.  ��� ���� (����)" << std::endl;
  std::cout << "7.  ��� ���� (�ε���)" << std::endl;
  std::cout << "8.  ��� ���� (��ü)" << std::endl;
  std::cout << "9.  ��� ����" << std::endl;
  std::cout << "10. ��� Ž�� (����)" << std::endl;
  std::cout << "11. ��� Ž�� (����, �ߺ�)" << std::endl;
  std::cout << "12. ��� Ž�� (����)" << std::endl;
  std::cout << "13. ��� Ž�� (����, �ߺ�)" << std::endl;
  std::cout << "14. ��� ���� (����)" << std::endl;
  std::cout << "15. ��� ���� (����)" << std::endl;
  std::cout << "16. ��� ���� (����)" << std::endl;
  std::cout << "17. ���α׷� ����" << std::endl;
}

void LinkedListConsole::Start() {
  Node** multiple_results;
  int menu_input, multiple_results_length;
  bool is_exit = false;

  while (!is_exit) {
    this->ClearConsole();
    this->ShowElements();
    this->ShowMenu();

    if (this->ReadFromConsole("�޴� �Է� : ", 1, &menu_input)) {
      this->ShowError();
      continue;
    }

    int data_input, *data_inputs;

    switch (menu_input) {
      case 1:
        this->ClearConsole();

        if (this->ReadFromConsole("�� �Է� : ", 1, &data_input)) {
          this->ShowError();
          break;
        }

        this->linked_list_->AppendFromHead(
            this->linked_list_->NewNode(data_input));

        break;
      case 2:
        this->ClearConsole();

        if (this->ReadFromConsole("�� �Է� : ", 1, &data_input)) {
          this->ShowError();
          break;
        }

        this->linked_list_->AppendFromTail(
            this->linked_list_->NewNode(data_input));

        break;
      case 3:
        this->ClearConsole();

        data_inputs = new int[2];

        if (this->ReadFromConsole("�ε����� �� �Է� : ", 2, data_inputs)) {
          this->ShowError();
          break;
        }

        this->linked_list_->InsertBefore(
            this->linked_list_->NewNode(data_inputs[1]), data_inputs[0]);

        delete data_inputs;
        break;
      case 4:
        this->ClearConsole();

        data_inputs = new int[2];

        if (this->ReadFromConsole("�ε����� �� �Է� : ", 2, data_inputs)) {
          this->ShowError();
          break;
        }

        this->linked_list_->InsertAfter(
            this->linked_list_->NewNode(data_inputs[1]), data_inputs[0]);

        delete data_inputs;
        break;
      case 5:
        this->ClearConsole();

        if (this->linked_list_->GetCount() == 0) {
          break;
        }

        this->linked_list_->DeleteFromHead();

        break;
      case 6:
        this->ClearConsole();

        if (this->linked_list_->GetCount() == 0) {
          break;
        }

        this->linked_list_->DeleteFromTail();

        break;
      case 7:
        this->ClearConsole();

        if (this->linked_list_->GetCount() == 0) {
          break;
        }

        if (this->ReadFromConsole("�ε��� �Է� : ", 1, &data_input)) {
          this->ShowError();
          break;
        }

        this->linked_list_->Delete(this->linked_list_->Read(data_input));

        break;
      case 8:
        this->ClearConsole();

        if (this->linked_list_->GetCount() == 0) {
          break;
        }

        this->linked_list_->DeleteAll();

        break;
      case 9:
        this->ClearConsole();

        data_inputs = new int[2];

        if (this->ReadFromConsole("�ε����� �� �Է� : ", 2, data_inputs)) {
          this->ShowError();
          break;
        }

        if (this->linked_list_->Read(data_inputs[0]) == nullptr) {
          this->ShowError();
        } else {
          this->linked_list_->Modify(this->linked_list_->Read(data_inputs[0]),
                                     data_inputs[1]);
        }

        delete data_inputs;

        break;
      case 10:
        this->ClearConsole();

        if (this->ReadFromConsole("�� �Է� : ", 1, &data_input)) {
          this->ShowError();
          break;
        }

        this->linked_list_->LinearSearchByUnique(data_input);

        std::cin.ignore();
        std::cin.get();

        break;
      case 11:
        this->ClearConsole();

        if (this->ReadFromConsole("�� �Է� : ", 1, &data_input)) {
          this->ShowError();
          break;
        }

        multiple_results_length = 0;

        this->linked_list_->LinearSearchByDuplicate(
            data_input, &multiple_results_length, &multiple_results);

        std::cin.ignore();
        std::cin.get();

        break;
      case 12:
        this->ClearConsole();

        if (this->ReadFromConsole("�� �Է� : ", 1, &data_input)) {
          this->ShowError();
        }

        if (!this->linked_list_->CheckListSorted()) {
          std::cout << "���ĵǾ� ���� �ʾƼ� Ž���� ������ �� �����ϴ�!"
                    << std::endl;
          std::cin.ignore();
          std::cin.get();
        }

        this->linked_list_->BinarySearchByUnique(data_input);

        std::cin.ignore();
        std::cin.get();

        break;
      case 13:
        this->ClearConsole();

        if (this->ReadFromConsole("�� �Է� : ", 1, &data_input)) {
          this->ShowError();
          break;
        }

        if (!this->linked_list_->CheckListSorted()) {
          std::cout << "���ĵǾ� ���� �ʾƼ� Ž���� ������ �� �����ϴ�!"
                    << std::endl;
          std::cin.ignore();
          std::cin.get();
          break;
        }

        multiple_results_length = 0;

        this->linked_list_->BinarySearchByDuplicate(
            data_input, &multiple_results_length, &multiple_results);

        std::cin.ignore();
        std::cin.get();

        break;
      case 14:
        this->ClearConsole();
        this->linked_list_->SortByBubble();
        break;
      case 15:
        this->ClearConsole();
        this->linked_list_->SortByInsertion();
        break;
      case 16:
        this->ClearConsole();
        this->linked_list_->SortBySelection();
        break;
      case 17:
        is_exit = true;
      default:
        std::cout << "�߸� �Է��ϼ̽��ϴ�! ���͸� ������ �ٽ� �����մϴ�!";
        std::cin.ignore();
        std::cin.get();
        break;
    }
  }
}

}  // namespace linked_list_console