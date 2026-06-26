// 데이터 제공자 팩토리. 환경변수로 백엔드 선택.
import type { DataProvider } from "./provider";
import { RestDataProvider } from "./restProvider";

let instance: DataProvider | null = null;

export function getDataProvider(): DataProvider {
  if (instance) return instance;
  const which = import.meta.env.VITE_DATA_PROVIDER ?? "rest";
  switch (which) {
    case "firebase":
      // const { FirebaseDataProvider } = await import("./firebaseProvider");
      // instance = new FirebaseDataProvider();
      throw new Error("Firebase provider 미구현 - restProvider 사용");
    case "rest":
    default:
      instance = new RestDataProvider();
  }
  return instance;
}

export type { DataProvider } from "./provider";
