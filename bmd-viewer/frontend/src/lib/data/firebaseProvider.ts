// Firebase 구현 자리 (향후 교체).
// 동일한 DataProvider 인터페이스를 구현하면 컴포넌트 변경 없이 전환 가능.
//
// 구현 시 매핑 가이드:
//   - 인증: Firebase Auth (signInWithEmailAndPassword)
//   - 환자/스터디: Firestore 컬렉션 (patients, studies, measurements)
//   - DICOM/파생 이미지: Firebase Storage
//   - 비동기 추론: Cloud Functions 또는 별도 추론 서비스 호출
//
// import type { DataProvider } from "./provider";
// export class FirebaseDataProvider implements DataProvider { ... }
export {};
