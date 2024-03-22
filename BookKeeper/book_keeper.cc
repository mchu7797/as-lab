/*
 * Created by Minseok Chu on 12/29/23.
 */

#include "book_keeper.h"

#include <iostream>

#ifdef WIN32
#include <optional>
#endif

#include "linked_list.h"

namespace book_keeper {

BookKeeper::BookKeeper() {
  this->transaction_list_ = new linked_list::SimpleList();
  this->transaction_count_ = 0;
}

std::optional<TransactionInfo> BookKeeper::Read(int id) {
  linked_list::Node* node = this->transaction_list_->BinarySearchByUnique(id);

  if (node == nullptr) {
    return std::nullopt;
  }

  TransactionInfo info;

  info.id = node->id;
  info.left_entry = node->left_entry;
  info.right_entry = node->right_entry;
  info.left_entry_comment = node->left_entry_comment;
  info.right_entry_comment = node->right_entry_comment;
  info.transaction_comment = node->transaction_comment;

  return info;
}

std::optional<TransactionInfo> BookKeeper::ReadByIndex(int index) {
  linked_list::Node* node = this->transaction_list_->Read(index);

  if (node == nullptr) {
    return std::nullopt;
  }

  TransactionInfo info;

  info.id = node->id;
  info.left_entry = node->left_entry;
  info.right_entry = node->right_entry;
  info.left_entry_comment = node->left_entry_comment;
  info.right_entry_comment = node->right_entry_comment;
  info.transaction_comment = node->transaction_comment;

  return info;
}

std::optional<Settlement> BookKeeper::GetSettlement() {
  book_keeper::Settlement settlement{0, 0};

  linked_list::Node* node = this->transaction_list_->Read(0);

  for (int i = 0; i < this->transaction_list_->GetCount(); ++i) {
    settlement.total_credit += node->left_entry;
    settlement.total_debit += node->right_entry;

    if (node->next == nullptr) {
      break;
    }

    node = node->next;
  }

  return settlement;
}

void BookKeeper::Create(TransactionInfo const& transactionInfo) {
  linked_list::Node* newNode = this->transaction_list_->NewNode(this->transaction_count_);
  ++transaction_count_;

  newNode->left_entry = transactionInfo.left_entry;
  newNode->right_entry = transactionInfo.right_entry;

  newNode->left_entry_comment = transactionInfo.left_entry_comment;
  newNode->right_entry_comment = transactionInfo.right_entry_comment;

  newNode->transaction_comment = transactionInfo.transaction_comment;

  this->transaction_list_->AppendFromTail(newNode);
  this->transaction_list_->SortBySelection();
}

errno_t BookKeeper::Update(book_keeper::TransactionInfo const& transactionInfo) {
  linked_list::Node* currentNode = this->transaction_list_->BinarySearchByUnique(transactionInfo.id);

  if (currentNode == nullptr) {
    // Cannot found transaction data
    return 1;
  }

  currentNode->left_entry = transactionInfo.left_entry;
  currentNode->right_entry = transactionInfo.right_entry;

  currentNode->left_entry_comment = transactionInfo.left_entry_comment;
  currentNode->right_entry_comment = transactionInfo.right_entry_comment;

  currentNode->transaction_comment = transactionInfo.transaction_comment;

  return 0;
}

errno_t BookKeeper::Delete(int id) {
  linked_list::Node* currentNode = this->transaction_list_->BinarySearchByUnique(id);

  if (currentNode == nullptr) {
    return 1;
  }

  if (this->transaction_list_->Delete(currentNode) == nullptr) {
    return 1;
  }

  return 0;
}

int BookKeeper::GetSize() {
  return transaction_list_->GetCount();
}

void BookKeeper::Sort() {
  this->transaction_list_->SortByInsertion();
}

}  // namespace book_keeper